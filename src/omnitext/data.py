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
