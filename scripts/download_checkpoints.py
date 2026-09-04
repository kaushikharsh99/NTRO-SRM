#!/usr/bin/env python3
"""Pretrained Model Checkpoint Downloader for NTRO-SRM.

Downloads and verifies model weights for:
1. SEN2SR-Lite (Default baseline CNN, ~0.47M parameters)
2. SEN2SR-Swin2SR (Higher-capacity Vision Transformer + MambaSR, ~12.9M parameters)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Pretrained model STAC URLs from Hugging Face tacofoundation/sen2sr
HF_SEN2SR_LITE_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
)
HF_SEN2SR_SWIN_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/main/mlm.json"
)


def download_variant(name: str, url: str, target_dir: Path) -> bool:
    """Download checkpoint assets via mlstac."""
    mlm_path = target_dir / "mlm.json"
    if mlm_path.is_file():
        print(f"[✓] {name} is already downloaded at: {target_dir}")
        return True

    print(f"[i] Downloading {name} from Hugging Face...")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        import mlstac
        mlstac.download(file=url, output_dir=str(target_dir))
        print(f"[✓] Successfully downloaded {name} to {target_dir}")
        return True
    except Exception as err:
        print(f"[!] Failed to download {name}: {err}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NTRO-SRM pretrained checkpoints.")
    parser.add_argument(
        "--model",
        choices=["all", "lite", "swin2sr"],
        default="all",
        help="Which model checkpoint to download (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoints_root = repo_root / "checkpoints"

    lite_dir = checkpoints_root / "SEN2SRLite"
    swin_dir = checkpoints_root / "SEN2SR"

    success = True
    if args.model in ("all", "lite"):
        ok = download_variant("SEN2SR-Lite", HF_SEN2SR_LITE_URL, lite_dir)
        success = success and ok

    if args.model in ("all", "swin2sr"):
        ok = download_variant("SEN2SR-Swin2SR", HF_SEN2SR_SWIN_URL, swin_dir)
        success = success and ok

    if success:
        print("\nAll requested model checkpoints are ready.")
        sys.exit(0)
    else:
        print("\nOne or more downloads failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
