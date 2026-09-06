#!/usr/bin/env python3
"""Side-by-side comparison: SEN2SR-Lite (base) vs SEN2SR-Lite FT (fine-tuned).

Runs both pipelines on the same input, reports quantitative deltas
(source-consistency RMSE each, inter-model MAE/RMSE/PSNR per band), and saves
RGB / CIR / zoom triptychs: [Input 10m | Lite 2.5m | Lite-FT 2.5m].

Usage:
  venv/bin/python scripts/compare_ft.py
  venv/bin/python scripts/compare_ft.py --input datasets/sample_s2/sample_s2_l2a.tif
  venv/bin/python scripts/compare_ft.py --device cpu --overlap 32
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rasterio  # noqa: E402

from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline  # noqa: E402
from ntro_srm.training.trainer import find_ft_checkpoint, load_checkpoint  # noqa: E402
from ntro_srm.utils.device import empty_device_cache, select_device  # noqa: E402
from ntro_srm.web.services.sr_service import (  # noqa: E402
    render_false_color_cir,
    render_true_color_rgb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base Lite vs fine-tuned Lite.")
    p.add_argument("--input", default="datasets/sample_s2/sample_s2_l2a.tif")
    p.add_argument("--output-dir", default="outputs/comparisons")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Default: auto (CUDA > MPS > CPU).")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--checkpoint", default=None,
                   help="FT/distilled .pt to compare. Default: auto (distill > FT).")
    return p.parse_args()


def build_lite_ft(device: torch.device, checkpoint: str | None = None) -> Sentinel2SRPipeline:
    pipeline = Sentinel2SRPipeline(model_variant="lite", device=device)
    ft_ckpt = Path(checkpoint) if checkpoint else find_ft_checkpoint(REPO_ROOT)
    if ft_ckpt is None:
        raise FileNotFoundError(
            "Fine-tuned weights not found. Run "
            "'python scripts/finetune_lite.py --epochs 5' first."
        )
    load_checkpoint(ft_ckpt, pipeline.model)
    pipeline.model.set_trainable(False)
    pipeline.model.eval()
    pipeline.model.model_variant = "lite-ft"
    print(f"  Fine-tuned weights: {ft_ckpt}")
    return pipeline


def source_consistency_rmse(sr: torch.Tensor, lr: torch.Tensor) -> float:
    """RMSE between area-downsampled SR and the observed LR (all bands)."""
    down = torch.nn.functional.interpolate(sr.unsqueeze(0), size=lr.shape[-2:], mode="area")
    return float(torch.sqrt(torch.mean((down - lr.unsqueeze(0)) ** 2)).item())


def inter_model_stats(a: torch.Tensor, b: torch.Tensor, bands: list[str]) -> list[tuple]:
    """Per-band MAE / RMSE / PSNR between two SR outputs (similarity, not accuracy)."""
    rows = []
    peak = max(float(a.max()), float(b.max()), 1e-6)
    for i, name in enumerate(bands):
        diff = (a[i] - b[i]).abs()
        mae = float(diff.mean())
        rmse = float(torch.sqrt((diff ** 2).mean()))
        psnr = 20 * np.log10(peak / (rmse + 1e-12))
        rows.append((name, mae, rmse, psnr))
    return rows


def main() -> int:
    args = parse_args()
    input_path = (REPO_ROOT / args.input).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"Input: {input_path}\nDevice: {device}")

    # 1. Base Lite
    print("\n[1/4] Running SEN2SR-Lite (base)...")
    empty_device_cache(device)
    gc.collect()
    t0 = time.perf_counter()
    pipe_lite = Sentinel2SRPipeline(model_variant="lite", device=device)
    res_lite = pipe_lite.predict(input_path=input_path,
                                 output_path=out_dir / "ft_compare_lite_2.5m.tif",
                                 overlap=args.overlap)
    lite_time = time.perf_counter() - t0
    print(f"  done in {lite_time:.2f}s, shape={tuple(res_lite.output_shape)}")
    del pipe_lite
    empty_device_cache(device)
    gc.collect()

    # 2. Fine-tuned Lite
    print("\n[2/4] Running SEN2SR-Lite FT (fine-tuned)...")
    t0 = time.perf_counter()
    pipe_ft = build_lite_ft(device, args.checkpoint)
    res_ft = pipe_ft.predict(input_path=input_path,
                             output_path=out_dir / "ft_compare_liteft_2.5m.tif",
                             overlap=args.overlap)
    ft_time = time.perf_counter() - t0
    print(f"  done in {ft_time:.2f}s, shape={tuple(res_ft.output_shape)}")

    sr_base = res_lite.sr_tensor.detach().cpu().float()
    sr_ft = res_ft.sr_tensor.detach().cpu().float()
    lr = res_ft.lr_tensor.detach().cpu().float()
    assert sr_base.shape == sr_ft.shape, f"{sr_base.shape} vs {sr_ft.shape}"

    # 3. Numbers
    print("\n[3/4] Metrics (reflectance units; PSNR = inter-model similarity)...")
    rmse_base = source_consistency_rmse(sr_base, lr)
    rmse_ft = source_consistency_rmse(sr_ft, lr)
    print(f"  Source-consistency RMSE  base Lite: {rmse_base:.6f}")
    print(f"  Source-consistency RMSE  FT Lite:   {rmse_ft:.6f} "
          f"({'better' if rmse_ft < rmse_base else 'worse'})")
    print(f"  {'Band':<6}{'MAE':>10}{'RMSE':>10}{'PSNR dB':>10}")
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for name, mae, rmse, psnr in inter_model_stats(sr_base, sr_ft, bands):
        print(f"  {name:<6}{mae:>10.6f}{rmse:>10.6f}{psnr:>10.2f}")

    # 4. Triptychs [Input 10m | Lite | Lite-FT]
    print("\n[4/4] Writing triptychs...")
    h, w = sr_base.shape[-2], sr_base.shape[-1]
    lr_t = lr.unsqueeze(0)
    lr_up = torch.nn.functional.interpolate(lr_t, size=(h, w), mode="nearest").squeeze(0).numpy()
    base_np = sr_base.numpy()
    ft_np = sr_ft.numpy()

    lr_rgb = np.transpose(lr_up[[2, 1, 0]], (1, 2, 0))
    base_rgb = np.transpose(base_np[[2, 1, 0]], (1, 2, 0))
    ft_rgb = np.transpose(ft_np[[2, 1, 0]], (1, 2, 0))
    lr_cir = np.transpose(lr_up[[6, 2, 1]], (1, 2, 0))
    base_cir = np.transpose(base_np[[6, 2, 1]], (1, 2, 0))
    ft_cir = np.transpose(ft_np[[6, 2, 1]], (1, 2, 0))

    lr_rgb_u8 = render_true_color_rgb(lr_rgb, ref_rgb=lr_rgb)
    base_rgb_u8 = render_true_color_rgb(base_rgb, ref_rgb=lr_rgb)
    ft_rgb_u8 = render_true_color_rgb(ft_rgb, ref_rgb=lr_rgb)
    lr_cir_u8 = render_false_color_cir(lr_cir, ref_cir=lr_cir)
    base_cir_u8 = render_false_color_cir(base_cir, ref_cir=lr_cir)
    ft_cir_u8 = render_false_color_cir(ft_cir, ref_cir=lr_cir)

    def triptych(left: np.ndarray, mid: np.ndarray, right: np.ndarray, sep: int = 10) -> np.ndarray:
        hh, ww, _ = left.shape
        canvas = np.zeros((hh, ww * 3 + sep * 2, 3), dtype=np.uint8)
        canvas[:, :ww] = left
        canvas[:, ww : ww + sep] = 255
        canvas[:, ww + sep : 2 * ww + sep] = mid
        canvas[:, 2 * ww + sep : 2 * ww + sep * 2] = 255
        canvas[:, 2 * ww + sep * 2 :] = right
        return canvas

    Image.fromarray(triptych(lr_rgb_u8, base_rgb_u8, ft_rgb_u8)).save(
        out_dir / "ft_comparison_rgb_triptych.png")
    Image.fromarray(triptych(lr_cir_u8, base_cir_u8, ft_cir_u8)).save(
        out_dir / "ft_comparison_cir_triptych.png")

    y0, x0, cs = 350, 450, 350
    Image.fromarray(triptych(lr_rgb_u8[y0 : y0 + cs, x0 : x0 + cs],
                             base_rgb_u8[y0 : y0 + cs, x0 : x0 + cs],
                             ft_rgb_u8[y0 : y0 + cs, x0 : x0 + cs], sep=8)).save(
        out_dir / "ft_comparison_zoom_triptych.png")

    # Amplified absolute-difference heatmap: where did fine-tuning change the
    # reconstruction? Mean |FT-base| over bands, scaled so the 99th percentile
    # saturates; the factor is printed and baked into the story, not hidden.
    diff = np.abs(ft_np - base_np).mean(axis=0)
    p99 = float(np.percentile(diff, 99)) + 1e-12
    amp = 1.0 / p99
    norm = np.clip(diff * amp, 0, 1)
    try:
        from matplotlib import colormaps
        heat = (np.asarray(colormaps["inferno"](norm))[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        heat = np.stack([gray, gray, gray], axis=-1)
    Image.fromarray(heat).save(out_dir / "ft_comparison_diff_heatmap.png")
    print(f"  diff heatmap x{amp:.1f} (p99-saturated) "
          f"-> {out_dir / 'ft_comparison_diff_heatmap.png'}")

    print(f"  {out_dir / 'ft_compare_lite_2.5m.tif'}")
    print(f"  {out_dir / 'ft_compare_liteft_2.5m.tif'}")
    print(f"  {out_dir / 'ft_comparison_rgb_triptych.png'}")
    print(f"  {out_dir / 'ft_comparison_cir_triptych.png'}")
    print(f"  {out_dir / 'ft_comparison_zoom_triptych.png'}")
    print(f"\n[SUCCESS] base={lite_time:.1f}s ft={ft_time:.1f}s on {device}")
    print("For HR-reference accuracy, also run:")
    print("  python -m ntro_srm.evaluation <sr.tif> --source <s2_10m.tif> "
          "--reference <hr_2.5m.tif>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
