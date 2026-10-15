"""Loading helpers for the OmniText-Bench benchmark.

OmniText-Bench layout (downloaded via ``scripts/download_data.py``)::

    OmniText-Bench/
        labels.json
        Input/<key>.png
        Application/
            Removal/GT/<key>.png
            Editing/{Ref1,Ref2}/{RefImage,GT}/<key>.png
            Insertion/{Ref1,Ref2}/{RefImage,GT}/<key>.png
            Repositioning/...
            Rescaling/...

``labels.json`` schema (per image key)::

    { "<key>": { "image": "Input/<key>.png",
                 "application": { "removal": {...}, "rescaling": {...},
                                  "repositioning": {...}, "editing": {...},
                                  "insertion": {...} } } }

Note on the manipulation pipeline: ``editing``, ``repositioning`` and ``rescaling``
operate on a *text-removed* image, so they consume the output of the removal stage
(``<removal_output_dir>/<key>/removal.png``) as their input, whereas ``removal`` and
``insertion`` operate on the original ``Input/<key>.png``.
"""
import json
from pathlib import Path

APPLICATIONS = ("removal", "editing", "insertion", "repositioning", "rescaling")

# Applications whose input image is the removal output rather than the original input.
USES_REMOVAL_OUTPUT = ("editing", "repositioning", "rescaling")

# application -> folder name under Application/ (mostly title-case of the key).
_APP_DIR = {
    "removal": "Removal",
    "editing": "Editing",
    "insertion": "Insertion",
    "repositioning": "Repositioning",
    "rescaling": "Rescaling",
}


def load_labels(dataset_root):
    """Load ``labels.json`` from the dataset root and return the dict."""
    labels_path = Path(dataset_root) / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.json not found at {labels_path}. "
            f"Run scripts/download_data.py to fetch OmniText-Bench first."
        )
    with open(labels_path) as f:
        return json.load(f)


def application_anns(labels, key, application):
    """Return the annotation sub-dict for one image key and application."""
    if application not in APPLICATIONS:
        raise ValueError(f"Unknown application {application!r}; expected one of {APPLICATIONS}")
    return labels[key]["application"][application]


def input_image_path(dataset_root, key):
    """Original input image: ``Input/<key>.png``."""
    return Path(dataset_root) / "Input" / f"{key}.png"


def removal_output_path(removal_output_dir, key):
    """Removal-stage output consumed by editing/repositioning/rescaling."""
    return Path(removal_output_dir) / key / "removal.png"


def ref_image_path(dataset_root, application, ref, key):
    """Reference image for style-based editing/insertion, e.g.
    ``Application/Editing/Ref1/RefImage/<key>.png``."""
    ref_dir = "Ref1" if ref in (1, "1", "ref1", "Ref1") else "Ref2"
    return Path(dataset_root) / "Application" / _APP_DIR[application] / ref_dir / "RefImage" / f"{key}.png"


def input_image_for(dataset_root, removal_output_dir, application, key):
    """Resolve the correct input image for an application, honouring the
    removal-first dependency for editing/repositioning/rescaling."""
    if application in USES_REMOVAL_OUTPUT:
        return removal_output_path(removal_output_dir, key)
    return input_image_path(dataset_root, key)


# External adapters intentionally do not read removal/editing ground-truth images.
_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_text_labels(path):
    """Read TextCtrl/ScenePair filename + whitespace + transcription records."""
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f'{path}:{number}: expected filename and nonempty text')
        name, text = parts
        if Path(name).name != name or name in result:
            raise ValueError(f'{path}:{number}: unsafe or duplicate filename {name!r}')
        result[name] = text.strip()
    if not result:
        raise ValueError(f'No text labels in {path}')
    return result


def load_external_samples(dataset_root, dataset_format, mask_dir=None, whole_crop=False):
    """Load native paths; masks are user-supplied, white=selected text.

    SCUT: root points to a split containing all_images/, not all_labels/ (GT).
    ScenePair: root contains i_s/, i_s.txt, i_t.txt. No t_f/ access is needed.
    """
    from PIL import Image

    root = Path(dataset_root)
    if dataset_format == 'scene-pair':
        source = read_text_labels(root / 'i_s.txt')
        target = read_text_labels(root / 'i_t.txt')
        if source.keys() != target.keys():
            raise ValueError('i_s.txt and i_t.txt must have identical filename sets')
        paths = [root / 'i_s' / name for name in sorted(source)]
    elif dataset_format == 'scut-enstext':
        paths = sorted(p for p in (root / 'all_images').iterdir()
                       if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        source = target = {}
    else:
        raise ValueError(f'Unknown external dataset format: {dataset_format}')
    if not paths:
        raise ValueError(f'No input images found in {root}')
    if whole_crop and dataset_format != 'scene-pair':
        raise ValueError('--whole-crop is only supported for ScenePair')
    masks = Path(mask_dir) if mask_dir else root / 'mask'
    samples = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            size = image.size
        # Mask files use the source stem with .png, regardless of source encoding.
        mask_path = masks / (path.stem + '.png')
        if mask_path.is_file():
            with Image.open(mask_path) as mask:
                if mask.size != size:
                    raise ValueError(f'Mask/image size mismatch: {mask_path}')
                if mask.convert('L').point(lambda v: 255 if v >= 128 else 0).getbbox() is None:
                    raise ValueError(f'Empty text mask: {mask_path}')
        elif not whole_crop:
            raise FileNotFoundError(f'Text mask required: {mask_path}. Supply --mask-dir with white-text PNG masks.')
        else:
            mask_path = None
        samples.append(dict(key=f'{index:06d}', image=str(path.resolve()),
                            mask=str(mask_path.resolve()) if mask_path else None,
                            name=path.name, size=size, source=source.get(path.name),
                            target=target.get(path.name)))
    return samples


def prepare_external_dataset(samples, destination):
    """Adapt to the fixed 512-square inference grid; retain original sizes for export."""
    from PIL import Image

    dest = Path(destination)
    labels = {}
    for sample in samples:
        key = sample['key']
        image = Image.open(sample['image']).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
        mask = (Image.open(sample['mask']).convert('L') if sample['mask']
                else Image.new('L', sample['size'], 255))
        mask = mask.resize((512, 512), Image.Resampling.NEAREST).point(lambda v: 255 if v >= 128 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError(f'Text mask vanished at inference resolution: {sample["name"]}')
        x0, y0, x1, y1 = bbox
        polygon = [x0, y0, x1-1, y0, x1-1, y1-1, x0, y1-1]
        ref_polygon = [v + 512 if i % 2 == 0 else v for i, v in enumerate(polygon)]
        for folder, content in [('Input', image), ('Masks', mask),
                                ('Application/Editing/Ref1/RefImage', image)]:
            path = dest / folder / f'{key}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            content.save(path)
        labels[key] = {'application': {
            'removal': {'polygon': polygon, 'mask': f'Masks/{key}.png'},
            'editing': {'ref1': {'text': {'reference_source': sample['source'], 'target': sample['target']},
                                  'target_polygon': polygon, 'ref_polygon': ref_polygon}}}}
    (dest / 'labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    return dest
