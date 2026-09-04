#!/usr/bin/env python3
"""CLI utility for end-to-end Sentinel-2 super-resolution using NTRO-SRM.

Usage:
    python scripts/sr_sentinel2.py \
        --input datasets/sample_s2/sample_s2_l2a.tif \
        --output outputs/sr_output.tif \
        --device cuda \
        --model lite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root src is on sys.path
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import rasterio
import torch

from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline, Sentinel2SRResult
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NTRO-SRM: Super-resolve Sentinel-2 Level-2A imagery to 2.5m GeoTIFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input Sentinel-2 GeoTIFF raster.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/sr_output.tif",
        help="Destination path for the 10-band 2.5m super-resolved GeoTIFF.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Compute device for neural network inference.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="lite",
        choices=["lite"],
        help="Model architecture variant.",
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        default="auto",
        choices=["auto", "s2_10000", "already_normalized"],
        help="Reflectance normalization strategy.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Sliding window tile overlap (pixels) for large images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.is_file():
        print(f"[ERROR] Input file does not exist: {input_path}")
        return 1

    print("=" * 75)
    print("NTRO-SRM: Sentinel-2 Super-Resolution Inference Pipeline")
    print("=" * 75)

    # 1. Inspect Input Raster
    with rasterio.open(input_path) as src:
        in_width = src.width
        in_height = src.height
        in_count = src.count
        in_crs = src.crs
        in_res = src.res
        in_bounds = src.bounds
        in_dtypes = src.dtypes

    print("INPUT DATASET:")
    print(f"  Path:            {input_path}")
    print(f"  Dimensions:      {in_width} x {in_height} (pixels)")
    print(f"  Band Count:      {in_count}")
    print(f"  Native Res:      {in_res[0]:.2f}m x {in_res[1]:.2f}m")
    print(f"  CRS:             {in_crs}")
    print(f"  Bounds:          left={in_bounds.left:.1f}, bottom={in_bounds.bottom:.1f}, "
          f"right={in_bounds.right:.1f}, top={in_bounds.top:.1f}")
    print(f"  Data Type:       {in_dtypes[0]}")
    print("-" * 75)

    # 2. Pipeline Initialization
    print("INITIALIZING MODEL:")
    print(f"  Model Variant:   SEN2SR-{args.model.capitalize()}")
    print(f"  Device:          {args.device}")
    if args.device == "cuda":
        print(f"  CUDA Device:     {torch.cuda.get_device_name(0)}")

    try:
        pipeline = Sentinel2SRPipeline(
            model_variant=args.model,
            device=args.device,
        )
    except Exception as err:
        print(f"[ERROR] Failed to load model: {err}")
        return 1

    print("-" * 75)

    # 3. Preprocessing & Inference Execution
    print("PROCESSING:")
    print(f"  Target Bands:    {S2_10BAND_NAMES}")
    print(f"  Normalization:   mode='{args.norm_mode}' (scale factor: 1/10000)")
    print("  Resampling:      Explicit bilinear resampling for 20m bands -> 10m grid")
    print(f"  Tiling Overlap:  {args.overlap} px")

    try:
        result: Sentinel2SRResult = pipeline.predict(
            input_path=input_path,
            output_path=output_path,
            normalization_mode=args.norm_mode,
            overlap=args.overlap,
        )
    except Exception as err:
        print(f"[ERROR] Pipeline execution failed: {err}")
        return 1

    lr_min = result.lr_tensor.min().item()
    lr_max = result.lr_tensor.max().item()
    sr_min = result.sr_tensor.min().item()
    sr_max = result.sr_tensor.max().item()

    print(f"  LR Tensor:       shape={result.input_shape}, range=[{lr_min:.4f}, {lr_max:.4f}]")
    print(f"  SR Tensor:       shape={result.output_shape}, range=[{sr_min:.4f}, {sr_max:.4f}]")
    print(f"  Inference Time:  {result.inference_time_ms:.2f} ms")
    if result.peak_gpu_memory_mb is not None:
        print(f"  Peak VRAM:       {result.peak_gpu_memory_mb:.2f} MB")
    print("-" * 75)

    # 4. Inspect Output Raster
    out_res_x = abs(result.output_transform.a)
    out_res_y = abs(result.output_transform.e)
    _, out_h, out_w = result.output_shape

    print("OUTPUT GEOTIFF:")
    print(f"  Path:            {result.output_path}")
    print(f"  Dimensions:      {out_w} x {out_h} (pixels)")
    print(f"  Band Count:      10")
    print(f"  SR Resolution:   {out_res_x:.2f}m x {out_res_y:.2f}m (~2.5m Ground Sampling Distance)")
    print(f"  Spatial Scale:   4x upscaling")
    print(f"  CRS:             {result.crs}")
    print(f"  Value Range:     [{sr_min:.4f}, {sr_max:.4f}]")
    print("=" * 75)
    print("STATUS: SUCCESS")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
