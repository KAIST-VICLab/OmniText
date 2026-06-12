"""Download the OmniText-Bench benchmark from Google Drive and verify its layout.

OmniText-Bench is released under a custom license (attribution required, source-file
redistribution prohibited). See the LICENSE.txt / TERMS.txt inside the download. This
script only fetches the official release; it does not redistribute the assets.

Usage:
    python scripts/download_data.py                 # -> ./OmniText-Bench
    python scripts/download_data.py --output-dir .  # choose where to extract
    python scripts/download_data.py --keep-zip      # keep the downloaded archive
"""
import argparse
import sys
import zipfile
from pathlib import Path

# Google Drive file id from the public release link:
# https://drive.google.com/file/d/16J8wyhpGcFnYwZLWa_01DLUOczIKgHVy/view
DRIVE_FILE_ID = "16J8wyhpGcFnYwZLWa_01DLUOczIKgHVy"

EXPECTED_DIRNAME = "OmniText-Bench"
EXPECTED_NUM_INPUTS = 150
EXPECTED_SUBDIRS = [
    "Input",
    "Application/Removal",
    "Application/Editing",
    "Application/Insertion",
    "Application/Repositioning",
    "Application/Rescaling",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=".", help="Directory to extract OmniText-Bench into (default: current dir).")
    p.add_argument("--zip-path", default="OmniText-Bench.zip", help="Where to save the downloaded archive.")
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded .zip after extraction.")
    p.add_argument("--force", action="store_true", help="Re-download even if the dataset already exists.")
    return p.parse_args()


def verify_layout(root: Path) -> bool:
    ok = True
    labels = root / "labels.json"
    if not labels.exists():
        print(f"  [MISSING] {labels}")
        ok = False
    for sub in EXPECTED_SUBDIRS:
        if not (root / sub).is_dir():
            print(f"  [MISSING] {root / sub}")
            ok = False
    inputs = root / "Input"
    if inputs.is_dir():
        n = len(list(inputs.glob("*.png")))
        if n != EXPECTED_NUM_INPUTS:
            print(f"  [WARN] expected {EXPECTED_NUM_INPUTS} input images, found {n}")
    return ok


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    target = out_dir / EXPECTED_DIRNAME

    if target.exists() and not args.force:
        print(f"{target} already exists. Verifying layout (use --force to re-download)...")
        sys.exit(0 if verify_layout(target) else 1)

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown (or pip install -r requirements.txt)")

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip_path)
    print(f"Downloading OmniText-Bench from Google Drive (id={DRIVE_FILE_ID}) ...")
    gdown.download(id=DRIVE_FILE_ID, output=str(zip_path), quiet=False)

    print(f"Extracting {zip_path} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    if not target.exists():
        # Archive may extract under a different top-level name; try to locate it.
        candidates = [p for p in out_dir.iterdir() if p.is_dir() and (p / "labels.json").exists()]
        if len(candidates) == 1 and candidates[0].name != EXPECTED_DIRNAME:
            candidates[0].rename(target)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)

    print("\nVerifying dataset layout...")
    ok = verify_layout(target)
    print("\nNOTE: OmniText-Bench is under a custom license (attribution required).")
    print(f"      Please read {target / 'LICENSE.txt'} and {target / 'TERMS.txt'} before use.")
    if ok:
        print(f"\nDone. Dataset ready at: {target}")
    else:
        sys.exit("\nDataset verification failed — see [MISSING] entries above.")


if __name__ == "__main__":
    main()
