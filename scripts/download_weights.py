"""Download and prepare all model weights OmniText depends on.

Fetches three Hugging Face repositories into the local HF cache and applies the one
required patch to the TextDiffuser-2 inpainting UNet config (``in_channels`` 4 -> 9),
which the original release required users to edit by hand.

Repositories:
  * stable-diffusion-v1-5/stable-diffusion-v1-5  (tokenizer + scheduler)
      Maintained community mirror of the original ``runwayml/stable-diffusion-v1-5``,
      which was removed from the Hugging Face Hub in 2024.
  * stabilityai/sd-vae-ft-mse                    (VAE)
  * JingyeChen22/textdiffuser2-full-ft-inpainting (text encoder + UNet) [patched]

Usage:
    python scripts/download_weights.py
"""
import argparse
import json
import sys
from pathlib import Path

SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
TEXTDIFFUSER2_REPO = "JingyeChen22/textdiffuser2-full-ft-inpainting"

# The inpainting UNet ships with in_channels=4 but is used with a 9-channel input
# (latent + masked-latent + mask). Patch the config so it loads with the right shape.
UNET_IN_CHANNELS = 9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--allow-patterns-only",
        action="store_true",
        help="Download only the subfolders OmniText loads (smaller footprint).",
    )
    return p.parse_args()


def patch_unet_in_channels(snapshot_dir: Path) -> None:
    config_path = snapshot_dir / "unet" / "config.json"
    if not config_path.exists():
        sys.exit(f"Expected UNet config not found at {config_path}")
    config = json.loads(config_path.read_text())
    current = config.get("in_channels")
    if current == UNET_IN_CHANNELS:
        print(f"  UNet config already has in_channels={UNET_IN_CHANNELS}; no patch needed.")
        return
    config["in_channels"] = UNET_IN_CHANNELS
    config_path.write_text(json.dumps(config, indent=2))
    print(f"  Patched {config_path}: in_channels {current} -> {UNET_IN_CHANNELS}")


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install -r requirements.txt")

    print(f"Downloading {SD15_REPO} (tokenizer + scheduler) ...")
    sd15_patterns = ["tokenizer/*", "scheduler/*", "model_index.json"] if args.allow_patterns_only else None
    snapshot_download(SD15_REPO, allow_patterns=sd15_patterns)

    print(f"Downloading {VAE_REPO} (VAE) ...")
    snapshot_download(VAE_REPO)

    print(f"Downloading {TEXTDIFFUSER2_REPO} (text encoder + UNet) ...")
    td2_patterns = ["text_encoder/*", "unet/*"] if args.allow_patterns_only else None
    td2_dir = Path(snapshot_download(TEXTDIFFUSER2_REPO, allow_patterns=td2_patterns))

    print("Applying TextDiffuser-2 UNet config patch ...")
    patch_unet_in_channels(td2_dir)

    print("\nAll weights downloaded and prepared. You can now run the application scripts.")


if __name__ == "__main__":
    main()
