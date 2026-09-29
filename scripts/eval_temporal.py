#!/usr/bin/env python3
"""Temporal vs single-image baseline: K-frame mean fusion vs K=1 on held-out tiles.

Usage:
  venv/bin/python scripts/eval_temporal.py
  venv/bin/python scripts/eval_temporal.py --split test --frames 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ntro_srm.models.sen2sr import SEN2SRModel  # noqa: E402
from ntro_srm.training.losses import SpectralSpatialLoss  # noqa: E402
from ntro_srm.training.temporal import TemporalS2Dataset, temporal_mean_lr  # noqa: E402
from ntro_srm.training.trainer import forward_native  # noqa: E402
from ntro_srm.utils.device import select_device  # noqa: E402


def collate(batch):
    return {"lr": torch.stack([b["lr"] for b in batch]),
            "hr": torch.stack([b["hr"] for b in batch]),
            "band_mask": torch.stack([b["band_mask"] for b in batch])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = select_device(args.device)
    ds = TemporalS2Dataset("datasets/paired/manifest.csv",
                           "datasets/raw_temporal/temporal_manifest.csv",
                           split=args.split, frames=args.frames)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
    print(f"temporal tiles ({args.split}, K={args.frames}): {len(ds)}")

    model = SEN2SRModel(model_variant="lite", device=str(device), auto_download=False)
    model.set_trainable(False)
    model.eval()
    loss_fn = SpectralSpatialLoss()
    acc_single = {k: 0.0 for k in ("pixel", "spectral", "gradient", "source_consistency")}
    acc_fused = {k: 0.0 for k in acc_single}
    n = 0
    with torch.no_grad():
        for batch in loader:
            lr, hr, mask = batch["lr"].to(device), batch["hr"].to(device), batch["band_mask"].to(device)
            pred_single = forward_native(model, lr[:, 0])
            pred_fused = forward_native(model, temporal_mean_lr(lr))
            for store, pred, src in ((acc_single, pred_single, lr[:, 0]),
                                     (acc_fused, pred_fused, temporal_mean_lr(lr))):
                comp = loss_fn.components(pred, hr, src, mask)
                for k in store:
                    store[k] += comp[k].item()
            n += 1
    print(f"K=1 single : " + " ".join(f"{k}={v / n:.6f}" for k, v in acc_single.items()))
    print(f"K={args.frames} mean  : " + " ".join(f"{k}={v / n:.6f}" for k, v in acc_fused.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
