#!/usr/bin/env python3
"""Verification script for NTRO-SRM SEN2SR adapter.

Validates:
1. Device detection (CUDA / CPU).
2. SEN2SR-Lite model initialization and pretrained checkpoint loading.
3. 10-band Sentinel-2 forward pass.
4. Verification of 4x spatial upscaling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path so script can be run directly
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch

try:
    from ntro_srm.models.sen2sr import SEN2SRModel
    from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
except ImportError as err:
    print(f"[FAIL] Could not import ntro_srm modules: {err}")
    sys.exit(1)


def main() -> int:
    print("=" * 70)
    print("NTRO-SRM Phase 1: SEN2SR-Lite Adapter Verification")
    print("=" * 70)

    # 1. Device check
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"CUDA Available:   {cuda_available}")
    if cuda_available:
        print(f"CUDA Device:      {torch.cuda.get_device_name(0)}")
    print(f"Target Device:    {device}")
    print("-" * 70)

    # 2. Model Initialization
    ckpt_dir = project_root / "checkpoints" / "SEN2SRLite"
    print(f"Loading SEN2SRModel (variant='lite') from: {ckpt_dir} ...")
    start_load = time.perf_counter()
    try:
        model = SEN2SRModel(
            model_variant="lite",
            device=device,
            checkpoint_dir=ckpt_dir,
            auto_download=True,
        )
    except Exception as err:
        print(f"[FAIL] Error initializing SEN2SRModel: {err}")
        return 1
    load_time = time.perf_counter() - start_load
    print(f"Model successfully loaded in {load_time:.2f}s")
    print("-" * 70)

    # 3. Create synthetic Sentinel-2 input tensor
    # Shape: (B=1, C=10, H=128, W=128)
    # Band ordering: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    # Dynamic range: [0.0, 1.0] realistic surface reflectance
    batch_size = 1
    num_bands = 10
    in_h, in_w = 128, 128
    torch.manual_seed(42)
    lr_tensor = torch.rand(batch_size, num_bands, in_h, in_w, dtype=torch.float32, device=device) * 0.4 + 0.05
    print(f"Input Band Names: {S2_10BAND_NAMES}")
    print(f"Input Shape:      {tuple(lr_tensor.shape)}")
    print(f"Input Dtype:      {lr_tensor.dtype}")
    print(f"Input Device:     {lr_tensor.device}")
    print(f"Input Min / Max:  {lr_tensor.min().item():.4f} / {lr_tensor.max().item():.4f}")
    print("-" * 70)

    # 4. Run prediction
    print("Running forward pass (predict)...")
    if cuda_available:
        torch.cuda.synchronize()
    start_infer = time.perf_counter()

    try:
        sr_tensor = model.predict(lr_tensor, auto_normalize=False, clamp_output=True)
    except Exception as err:
        print(f"[FAIL] Inference execution failed: {err}")
        return 1

    if cuda_available:
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - start_infer

    # 5. Report results
    print(f"Output Shape:     {tuple(sr_tensor.shape)}")
    print(f"Output Dtype:     {sr_tensor.dtype}")
    print(f"Output Device:    {sr_tensor.device}")
    print(f"Output Min / Max: {sr_tensor.min().item():.4f} / {sr_tensor.max().item():.4f}")
    print(f"Execution Time:   {infer_time * 1000:.2f} ms")
    print("-" * 70)

    # 6. Verify 4x spatial upscaling and channel count
    expected_out_h = in_h * 4
    expected_out_w = in_w * 4
    expected_shape = (batch_size, num_bands, expected_out_h, expected_out_w)

    shape_ok = tuple(sr_tensor.shape) == expected_shape
    finite_ok = bool(torch.isfinite(sr_tensor).all().item())
    nonneg_ok = bool((sr_tensor >= 0.0).all().item())

    print(f"Validation Checks:")
    print(f"  [1] Output shape matches 4x upscaling ({expected_shape}): {'PASS' if shape_ok else 'FAIL'}")
    print(f"  [2] Output contains finite values (no NaNs/Infs):         {'PASS' if finite_ok else 'FAIL'}")
    print(f"  [3] Non-negative reflectance constraint (min >= 0.0):     {'PASS' if nonneg_ok else 'FAIL'}")
    print("-" * 70)

    if shape_ok and finite_ok and nonneg_ok:
        print("OVERALL STATUS: PASS")
        return 0
    else:
        print("OVERALL STATUS: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
