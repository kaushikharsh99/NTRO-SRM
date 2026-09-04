#!/usr/bin/env python3
"""Visualization tool for Sentinel-2 super-resolution inspection.

Generates side-by-side comparisons of True-Color (RGB) and False-Color (NIR/Red/Green)
between low-resolution (10m) and super-resolved (2.5m) imagery.

Usage:
    python scripts/visualize_sr.py \
        --lr datasets/sample_s2/sample_s2_l2a.tif \
        --sr outputs/sr_output.tif \
        --output outputs/sr_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root src is on sys.path
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from ntro_srm.data.sentinel2 import Sentinel2Reader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison visualizations for Sentinel-2 super-resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lr",
        type=str,
        required=True,
        help="Path to low-resolution input Sentinel-2 GeoTIFF.",
    )
    parser.add_argument(
        "--sr",
        type=str,
        required=True,
        help="Path to super-resolved 2.5m output GeoTIFF.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/sr_comparison.png",
        help="Path to save the comparison visualization image.",
    )
    parser.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=[2.0, 98.0],
        help="Lower and upper percentiles for display contrast stretching.",
    )
    return parser.parse_args()


def stretch_rgb(arr: np.ndarray, ref_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Photorealistic True Color stretch with shared radiometric bounds."""
    if ref_arr is None:
        ref_arr = arr
    valid = np.isfinite(ref_arr)
    floor = min(max(0.0, float(np.percentile(ref_arr[valid], 1.0))), 0.03)
    p99 = float(np.percentile(ref_arr[valid], 99.0))
    ceiling = max(p99, 0.28)
    denom = ceiling - floor if ceiling > floor else 1.0
    norm = np.clip((arr - floor) / denom, 0.0, 1.0)
    return np.power(norm, 1.0 / 1.9)



def main() -> int:
    args = parse_args()
    lr_path = Path(args.lr).resolve()
    sr_path = Path(args.sr).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not lr_path.is_file():
        print(f"[ERROR] LR file not found: {lr_path}")
        return 1
    if not sr_path.is_file():
        print(f"[ERROR] SR file not found: {sr_path}")
        return 1

    # 1. Read LR data (use Sentinel2Reader to ensure standardized 10-band stack)
    reader = Sentinel2Reader(lr_path)
    lr_data = reader.read()
    # (10, H, W)
    lr_tensor = lr_data.tensor.float()
    if lr_tensor.max() > 1.5:
        lr_tensor = lr_tensor / 10000.0
    lr_np = lr_tensor.numpy()

    # 2. Read SR data (from write_sr_geotiff output)
    with rasterio.open(sr_path) as src:
        sr_np = src.read()  # (10, H_sr, W_sr)

    # Band indexing in standard 10-band stack:
    # 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red), 6: B08 (NIR)
    # RGB True Color: Red=2, Green=1, Blue=0
    lr_rgb = np.transpose(lr_np[[2, 1, 0], :, :], (1, 2, 0))
    sr_rgb = np.transpose(sr_np[[2, 1, 0], :, :], (1, 2, 0))

    # False Color (Color Infrared): Red=B08 (6), Green=B04 (2), Blue=B03 (1)
    lr_cir = np.transpose(lr_np[[6, 2, 1], :, :], (1, 2, 0))
    sr_cir = np.transpose(sr_np[[6, 2, 1], :, :], (1, 2, 0))

    # Apply percentile stretching for display
    p_low, p_high = args.percentiles
    lr_rgb_disp = stretch_rgb(lr_rgb, p_low, p_high)
    sr_rgb_disp = stretch_rgb(sr_rgb, p_low, p_high)
    lr_cir_disp = stretch_rgb(lr_cir, p_low, p_high)
    sr_cir_disp = stretch_rgb(sr_cir, p_low, p_high)

    # 3. Create figure with overview and zoomed crop
    fig = plt.figure(figsize=(16, 14), dpi=200)
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(lr_rgb_disp)
    ax1.set_title(f"1. Low-Resolution (LR) True Color (RGB)\nNative 10m GSD — {lr_rgb.shape[1]}x{lr_rgb.shape[0]} px",
                  fontsize=13, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(sr_rgb_disp)
    ax2.set_title(f"2. Super-Resolved (SR) True Color (RGB)\nSEN2SR-Lite ~2.5m GSD — {sr_rgb.shape[1]}x{sr_rgb.shape[0]} px (4x)",
                  fontsize=13, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(lr_cir_disp)
    ax3.set_title(f"3. Low-Resolution (LR) False Color (NIR / Red / Green)\nNative 10m GSD — Bands [B08, B04, B03]",
                  fontsize=13, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(sr_cir_disp)
    ax4.set_title(f"4. Super-Resolved (SR) False Color (NIR / Red / Green)\nSEN2SR-Lite ~2.5m GSD — Bands [B08, B04, B03] (4x)",
                  fontsize=13, fontweight="bold")
    ax4.axis("off")

    fig.suptitle("NTRO-SRM: Sentinel-2 Multi-Spectral Super-Resolution Mapping (Phase 2)",
                 fontsize=16, fontweight="bold", y=0.96)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] Comparison visualization written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
