"""SIH Round-2: inference-optimization benchmarks (legacy vs optimized).

Compares wall-clock time, tile counts, and numerical agreement between the
upstream single-tile hard-crop path and the optimized batched weighted-blend
path. Writes a Markdown table suitable for the SIH presentation.

Usage:
    venv/bin/python scripts/benchmark_inference.py --size 256 --repeats 2
    venv/bin/python scripts/benchmark_inference.py --use-sample --repeats 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch

from ntro_srm.inference.tiled import TiledInferenceConfig, compute_tile_grid


def _load_model(device: str = "cpu"):
    from ntro_srm.models.sen2sr import SEN2SRModel

    ckpt = project_root / "checkpoints" / "SEN2SRLite"
    return SEN2SRModel(model_variant="lite", device=device, checkpoint_dir=ckpt, auto_download=False)


def _synthetic(h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 10, h, w, generator=g, dtype=torch.float32) * 0.45 + 0.05
    # Add an edge so seam behaviour is exercised.
    base[:, :, h // 3 : 2 * h // 3, :] *= 1.6
    return base.clamp(0.0, 1.0)


def _timed(fn, repeats: int = 1) -> tuple[torch.Tensor, float]:
    # Warmup (kernel autotune, lazy init) excluded from timing.
    out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, sum(times) / max(1, len(times))


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark legacy vs optimized SEN2SR inference.")
    ap.add_argument("--size", type=int, default=256, help="Synthetic square size (ignored with --use-sample).")
    ap.add_argument("--rect", type=str, default="", help="Optional rectangular HxW, e.g. 192x320.")
    ap.add_argument("--use-sample", action="store_true", help="Use datasets/sample_s2/sample_s2_l2a.tif (256x256).")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output", type=str, default="benchmarks/inference_cpu.md")
    args = ap.parse_args()

    if args.rect:
        h, w = (int(v) for v in args.rect.lower().split("x"))
        lr = _synthetic(h, w)
        label = f"synthetic {h}x{w}"
    elif args.use_sample:
        from ntro_srm.data.sentinel2 import Sentinel2Reader
        from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a

        sample = project_root / "datasets" / "sample_s2" / "sample_s2_l2a.tif"
        raster = Sentinel2Reader(sample).read()
        lr = normalize_sentinel2_l2a(raster.tensor, mode="auto", nodata_value=raster.nodata).unsqueeze(0)
        h, w = lr.shape[-2:]
        label = f"sample {h}x{w}"
    else:
        h = w = int(args.size)
        lr = _synthetic(h, w)
        label = f"synthetic {h}x{w}"

    model = _load_model(args.device)
    model.eval()

    legacy_tiles = len(compute_tile_grid(h, w, 128, 32))
    print(f"Input: {label} | legacy tiles (unique grid): {legacy_tiles}")

    out_legacy, t_legacy = _timed(
        lambda: model.predict(lr, overlap=32, use_legacy_tiling=True), args.repeats
    )
    out_opt1, t_opt1 = _timed(
        lambda: model.predict(lr, overlap=32, batch_size=1, blend_mode="linear"), args.repeats
    )
    out_opt4, t_opt4 = _timed(
        lambda: model.predict(lr, overlap=32, batch_size=args.batch_size, blend_mode="linear"), args.repeats
    )
    out_avg, t_avg = _timed(
        lambda: model.predict(lr, overlap=32, batch_size=args.batch_size, blend_mode="average"), args.repeats
    )

    def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        d = (a.float() - b.float()).abs()
        return float(d.max()), float(d.mean())

    max_l1, mean_l1 = _diff(out_legacy, out_opt1)
    max_l4, mean_l4 = _diff(out_legacy, out_opt4)

    rows = [
        ("legacy (1/tile, hard-crop)", t_legacy, 1.0, "-", "-"),
        ("optimized batch=1 linear-blend", t_opt1, t_legacy / max(t_opt1, 1e-9), f"{max_l1:.2e}", f"{mean_l1:.2e}"),
        (f"optimized batch={args.batch_size} linear-blend", t_opt4, t_legacy / max(t_opt4, 1e-9), f"{max_l4:.2e}", f"{mean_l4:.2e}"),
        (f"optimized batch={args.batch_size} average-blend", t_avg, t_legacy / max(t_avg, 1e-9), "-", "-"),
    ]
    md = [
        "# Inference benchmark (legacy vs optimized)",
        "",
        f"Input: {label} | device={args.device} | repeats={args.repeats} | tile=128 overlap=32",
        "",
        "| path | mean ms | speedup | max|legacy-opt| | mean|.| |",
        "|---|---|---|---|---|",
    ]
    for name, ms, sp, mx, mn in rows:
        md.append(f"| {name} | {ms:.1f} | {sp if isinstance(sp, str) else f'{sp:.2f}x'} | {mx} | {mn} |")
    md += [
        "",
        "Notes: legacy = upstream single-tile hard-crop path (`use_legacy_tiling=True`); "
        "optimized batches tiles and uses partition-of-unity blending. Small numerical "
        "differences vs legacy are expected at overlap seams (that is the seam fix).",
        "",
    ]
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
