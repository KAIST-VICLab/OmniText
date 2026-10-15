"""Run existing OmniText removal/editing on SCUT-EnsText or ScenePair inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from omnitext.data import load_external_samples, prepare_external_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-format', choices=['scut-enstext', 'scene-pair'], required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, help='PNG masks named <source-stem>.png; white=text')
    parser.add_argument('--whole-crop', action='store_true', help='ScenePair only: explicitly select the entire crop if no mask exists')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--removal-config', default='configs/removal.yaml')
    parser.add_argument('--editing-config', default='configs/editing.yaml')
    parser.add_argument('--gpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and report plan without torch, weights or GPU')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    return args


def run(args):
    from PIL import Image

    samples = load_external_samples(args.dataset_root, args.dataset_format,
                                    args.mask_dir, args.whole_crop)
    if args.limit:
        samples = samples[:args.limit]
    if len({Path(s['name']).stem for s in samples}) != len(samples):
        raise ValueError('Input stems must be unique for masks and PNG output naming')
    # Both configs are checked before a GPU process is started.
    configs = [args.removal_config]
    if args.dataset_format == 'scene-pair':
        configs.append(args.editing_config)
    for config in configs:
        if not Path(config).is_file():
            raise FileNotFoundError(config)
    print(json.dumps({'dataset_format': args.dataset_format, 'count': len(samples),
                      'stages': ['removal', 'editing'] if len(configs) == 2 else ['removal'],
                      'resolution': [512, 512], 'samples': samples}, indent=2))
    # Preparation is also exercised by dry-run, including mask resampling validation.
    with tempfile.TemporaryDirectory(prefix='omnitext-') as work:
        root = prepare_external_dataset(samples, Path(work) / 'dataset')
        if args.dry_run:
            return
        scripts = Path(__file__).resolve().parent
        removal = Path(work) / 'removal'
        editing = Path(work) / 'editing'
        common = ['--dataset-root', str(root)]
        if args.gpu is not None:
            common += ['--gpu', args.gpu]
        subprocess.run([sys.executable, str(scripts / 'run_removal.py'),
                        '--config', args.removal_config, '--output-dir', str(removal), *common], check=True)
        if args.dataset_format == 'scene-pair':
            subprocess.run([sys.executable, str(scripts / 'run_editing.py'),
                            '--config', args.editing_config, '--output-dir', str(editing),
                            '--removal-output-dir', str(removal), '--ref', 'ref1', *common], check=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            path = (editing / sample['key'] / '00.png' if len(configs) == 2
                    else removal / sample['key'] / 'removal.png')
            # PNG avoids adding JPEG artifacts; native size is restored for alignment.
            output = args.output_dir / (Path(sample['name']).stem + '.png')
            with Image.open(path) as image:
                image.resize(sample['size'], Image.Resampling.LANCZOS).save(output)
        (args.output_dir / 'manifest.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run(parse_args())
