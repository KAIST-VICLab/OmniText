"""Download the OmniText-Bench benchmark from the Hugging Face Hub and verify its layout.

Dataset: https://huggingface.co/datasets/agusgun/OmniText-Bench

OmniText-Bench is released under a custom license (attribution required, source-file
redistribution prohibited). See the LICENSE.txt / TERMS.txt inside the download. This
script only fetches the official release; it does not redistribute the assets.

The dataset is *gated*: before downloading you must
    1. open the dataset page above and accept the license terms, and
    2. authenticate, either via `huggingface-cli login` or by passing a token
       (`--token hf_...` or the HF_TOKEN environment variable).

Usage:
    python scripts/download_data.py                 # -> ./OmniText-Bench
    python scripts/download_data.py --output-dir .  # choose where to download
    python scripts/download_data.py --token hf_...  # explicit access token
"""
import argparse
import os
import sys
from pathlib import Path

HF_REPO_ID = "agusgun/OmniText-Bench"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

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
    p.add_argument("--output-dir", default=".", help="Directory to place OmniText-Bench in (default: current dir).")
    p.add_argument("--token", default=None,
                   help="Hugging Face access token (defaults to HF_TOKEN env var or the cached login).")
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
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub (or pip install -r requirements.txt)")

    token = args.token or os.environ.get("HF_TOKEN") or None  # empty string -> use cached login

    print(f"Downloading OmniText-Bench from {HF_DATASET_URL} ...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(target),
            token=token,
            force_download=args.force,
        )
    except GatedRepoError:
        sys.exit(
            "\nAccess denied: OmniText-Bench is a gated dataset.\n"
            f"  1. Accept the license terms at {HF_DATASET_URL}\n"
            "  2. Authenticate with `huggingface-cli login`, or pass --token / set HF_TOKEN\n"
        )
    except RepositoryNotFoundError:
        sys.exit(
            f"\nRepository {HF_REPO_ID} not found or not accessible.\n"
            "If you are not logged in, run `huggingface-cli login` first (the dataset is gated).\n"
        )

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
