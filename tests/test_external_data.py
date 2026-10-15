import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
from PIL import Image
from omnitext import data


class ExternalDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def image(self, path, color='white', size=(80, 40)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', size, color).save(path)

    def scene(self):
        self.image(self.root / 'i_s/a.jpg')
        (self.root / 'i_s.txt').write_text('a.jpg Old text\n')
        (self.root / 'i_t.txt').write_text('a.jpg New text\n')

    def test_scene_pair_without_gt_and_pair_by_name(self):
        self.scene()
        self.image(self.root / 'i_s/b.png')
        (self.root / 'i_s.txt').write_text('a.jpg Old text\nb.png Second\n')
        (self.root / 'i_t.txt').write_text('b.png Other\na.jpg New text\n')
        samples = data.load_external_samples(self.root, 'scene-pair', whole_crop=True)
        self.assertEqual(samples[0]['target'], 'New text')
        prepared = data.prepare_external_dataset(samples, self.root / 'prepared')
        labels = data.load_labels(prepared)
        ann = labels['000000']['application']['editing']['ref1']
        self.assertEqual(ann['text']['reference_source'], 'Old text')
        self.assertEqual(ann['ref_polygon'], [512, 0, 1023, 0, 1023, 511, 512, 511])
        with Image.open(data.input_image_path(prepared, '000000')) as image:
            self.assertEqual(image.size, (512, 512))

    def test_scut_masks_and_no_gt_access(self):
        self.image(self.root / 'all_images/one.JPG')
        self.image(self.root / 'mask/one.png', 'black')
        with self.assertRaisesRegex(ValueError, 'Empty'):
            data.load_external_samples(self.root, 'scut-enstext')
        self.image(self.root / 'mask/one.png')
        samples = data.load_external_samples(self.root, 'scut-enstext')
        self.assertEqual(len(samples), 1)
        self.assertIsNone(samples[0]['target'])
        self.image(self.root / 'mask/one.png', size=(8, 8))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            data.load_external_samples(self.root, 'scut-enstext')

    def test_missing_masks_and_bad_labels(self):
        self.scene()
        with self.assertRaises(FileNotFoundError):
            data.load_external_samples(self.root, 'scene-pair')
        (self.root / 'i_t.txt').write_text('b.jpg New\n')
        with self.assertRaisesRegex(ValueError, 'identical'):
            data.load_external_samples(self.root, 'scene-pair', whole_crop=True)
        (self.root / 'i_t.txt').write_text('../a.jpg New\n')
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            data.read_text_labels(self.root / 'i_t.txt')
        (self.root / 'i_t.txt').write_text('a.jpg New\na.jpg Again\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            data.read_text_labels(self.root / 'i_t.txt')

    def test_existing_benchmark_paths(self):
        (self.root / 'labels.json').write_text(json.dumps({'original': {'application': {'removal': {'polygon': [1, 2]}}}}))
        self.assertEqual(data.application_anns(data.load_labels(self.root), 'original', 'removal'), {'polygon': [1, 2]})
        self.assertEqual(data.removal_output_path('out', 'original'), Path('out/original/removal.png'))
        self.assertEqual(data.ref_image_path(self.root, 'editing', 'ref2', 'original'), self.root / 'Application/Editing/Ref2/RefImage/original.png')

    def test_orchestration_and_export(self):
        self.scene()
        spec = importlib.util.spec_from_file_location('external_cli', 'scripts/run_external.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = cli.parse_args(['--dataset-format', 'scene-pair', '--dataset-root', str(self.root),
                              '--whole-crop', '--output-dir', str(self.root / 'out')])
        calls = []

        def fake_gpu(command, check):
            calls.append(command)
            out = Path(command[command.index('--output-dir') + 1])
            self.image(out / '000000' / ('removal.png' if len(calls) == 1 else '00.png'), size=(512, 512))
            if len(calls) == 2:
                removal = Path(command[command.index('--removal-output-dir') + 1])
                self.assertTrue((removal / '000000/removal.png').is_file())

        with patch.object(cli.subprocess, 'run', side_effect=fake_gpu), patch('builtins.print'):
            cli.run(args)
        self.assertEqual(len(calls), 2)
        with Image.open(self.root / 'out/a.png') as image:
            self.assertEqual(image.size, (80, 40))
        self.assertEqual(json.loads((self.root / 'out/manifest.json').read_text())[0]['name'], 'a.jpg')

    def test_cpu_cli_dry_run(self):
        self.scene()
        result = subprocess.run([sys.executable, 'scripts/run_external.py', '--dataset-format', 'scene-pair',
            '--dataset-root', str(self.root), '--whole-crop', '--dry-run', '--output-dir', str(self.root / 'out')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['stages'], ['removal', 'editing'])
        self.assertFalse((self.root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
