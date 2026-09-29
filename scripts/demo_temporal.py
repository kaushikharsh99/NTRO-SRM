#!/usr/bin/env python3
"""Full-scene temporal demo: single (ref frame) vs K-mean vs learned fusion.

Writes GeoTIFFs + RGB triptych + amplified fused-vs-single heatmap for one site.
Usage:
  venv/bin/python scripts/demo_temporal.py --site IND_DELHI_URBAN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import rasterio  # noqa: E402
from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline  # noqa: E402
from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a  # noqa: E402
from ntro_srm.training.fusion import TemporalFusionHead  # noqa: E402
from ntro_srm.training.trainer import find_ft_checkpoint, load_checkpoint  # noqa: E402
from ntro_srm.training.wald import wald_degrade  # noqa: E402
from ntro_srm.utils.device import select_device  # noqa: E402
from ntro_srm.utils.geotiff import write_sr_geotiff  # noqa: E402
from ntro_srm.web.services.sr_service import render_true_color_rgb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="IND_DELHI_URBAN")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = ap.parse_args()
    device = select_device(args.device)
    out_dir = REPO_ROOT / "outputs" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    manifest = REPO_ROOT / "datasets" / "raw_temporal" / "temporal_manifest.csv"
    rows = [r for r in csv.DictReader(manifest.open()) if r["site_id"] == args.site]
    rows = sorted(rows, key=lambda r: float(r.get("cloud_pct") or 99))[:3]
    assert len(rows) == 3, f"need 3 temporal chips for {args.site}"
    print("dates:", [r["date"][:10] for r in rows])

    pipe = Sentinel2SRPipeline(model_variant="lite", device=device)
    ft = find_ft_checkpoint(REPO_ROOT)
    load_checkpoint(ft, pipe.model)
    pipe.model.set_trainable(False)
    pipe.model.eval()

    lrs, srs, ref_meta = [], [], None
    for r in rows:
        with rasterio.open(REPO_ROOT / r["path"]) as src:
            arr = torch.from_numpy(src.read().astype(np.float32))
            if ref_meta is None:
                ref_meta = (src.transform, src.crs)
        lr10 = normalize_sentinel2_l2a(arr, mode="auto", nodata_value=None)
        lrs.append(lr10)
        # Full-scene SR via pipeline tiling on a synthetic норме tile is heavy;
        # use Wald-degraded proxy: degrade each date to 40m, SR back to 10m grid.
        with torch.no_grad():
            sr = pipe.model.predict(wald_degrade(lr10, 4).unsqueeze(0).to(device),
                                    auto_normalize=False, clamp_output=True, overlap=32)
        srs.append(sr.squeeze(0).cpu())
        print(f"  {r['date'][:10]} sr done {tuple(sr.shape)}")
    lr_stack = torch.stack(lrs)
    sr_stack = torch.stack(srs)
    mean_sr = sr_stack.mean(dim=0)

    head = TemporalFusionHead(frames=3)
    head.load_state_dict(torch.load(REPO_ROOT / "outputs/finetune/fusion_head.pt",
                                    map_location="cpu", weights_only=False)["head_state"])
    head.to(device)
    head.eval()
    with torch.no_grad():
        fused = head(sr_stack.unsqueeze(0).to(device), lr_stack.unsqueeze(0).to(device)).squeeze(0).cpu()
    print("fused:", tuple(fused.shape))

    transform, crs = ref_meta
    from rasterio.transform import Affine
    out_t = Affine(transform.a / 4, transform.b, transform.c, transform.d, transform.e / 4, transform.f)
    stem = args.site.lower()
    paths = {}
    for name, tensor in (("single", srs[0]), ("mean", mean_sr), ("fused", fused)):
        p = out_dir / f"temporal_{stem}_{name}_10m.tif"
        write_sr_geotiff(output_path=p, tensor=tensor, transform=out_t, crs=crs,
                         model_name=f"temporal-{name}", input_gsd="40m", output_gsd="10m",
                         upscale_factor=4)
        paths[name] = tensor.numpy()
        print("wrote", p.name)

    def rgb(x):
        return render_true_color_rgb(np.transpose(x[[2, 1, 0]], (1, 2, 0)),
                                     ref_rgb=np.transpose(paths["single"][[2, 1, 0]], (1, 2, 0)))

    panels = [rgb(paths["single"]), rgb(paths["mean"]), rgb(paths["fused"])]
    h, w, _ = panels[0].shape
    trip = np.zeros((h, w * 3 + 20, 3), np.uint8)
    trip[:, :w] = panels[0]
    trip[:, w:w + 10] = 255
    trip[:, w + 10:2 * w + 10] = panels[1]
    trip[:, 2 * w + 10:2 * w + 20] = 255
    trip[:, 2 * w + 20:] = panels[2]
    Image.fromarray(trip).save(out_dir / f"temporal_{stem}_triptych.png")

    diff = np.abs(paths["fused"] - paths["single"]).mean(axis=0)
    p99 = float(np.percentile(diff, 99)) + 1e-12
    norm = np.clip(diff / p99, 0, 1)
    try:
        from matplotlib import colormaps
        heat = (np.asarray(colormaps["inferno"](norm))[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        heat = np.stack([gray, gray, gray], -1)
    Image.fromarray(heat).save(out_dir / f"temporal_{stem}_diff_heatmap.png")
    print(f"[SUCCESS] triptych + diff heatmap (x{1 / p99:.1f}) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
