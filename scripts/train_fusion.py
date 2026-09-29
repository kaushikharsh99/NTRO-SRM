#!/usr/bin/env python3
"""Train/evaluate the temporal fusion head (frozen Lite-FT backbone).

1. Cache per-frame SR stacks once (datasets/cache/temporal_sr/, gitignored).
2. Train the ~10k-param attention head on train temporal tiles.
3. Report K=1 vs K-mean vs K-fused on val/test with SpectralSpatialLoss parts.

Usage:
  venv/bin/python scripts/train_fusion.py --epochs 40
  venv/bin/python scripts/train_fusion.py --eval-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ntro_srm.models.sen2sr import SEN2SRModel  # noqa: E402
from ntro_srm.training.fusion import TemporalFusionHead  # noqa: E402
from ntro_srm.training.losses import SpectralSpatialLoss  # noqa: E402
from ntro_srm.training.temporal import TemporalS2Dataset, temporal_mean_lr  # noqa: E402
from ntro_srm.training.trainer import find_ft_checkpoint, forward_native, load_checkpoint  # noqa: E402
from ntro_srm.utils.device import select_device  # noqa: E402

CACHE_DIR = REPO_ROOT / "datasets" / "cache" / "temporal_sr"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--output", default="outputs/finetune/fusion_head.pt")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


class CachedTemporal(Dataset):
    def __init__(self, split: str, frames: int) -> None:
        ds = TemporalS2Dataset("datasets/paired/manifest.csv",
                               "datasets/raw_temporal/temporal_manifest.csv",
                               split=split, frames=frames)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.items: list[dict] = []
        for i in range(len(ds)):
            s = ds[i]
            key = f"{split}_{s['meta']['site_id']}_{s['meta']['y']}_{s['meta']['x']}.pt"
            self.items.append({"key": key, "lr": s["lr"], "hr": s["hr"],
                               "band_mask": s["band_mask"]})

    def __len__(self) -> int:
        return len(self.items)


def build_cache(backbone: SEN2SRModel, device: torch.device, frames: int) -> None:
    backbone.eval()
    for split in ("train", "val", "test"):
        try:
            ds = CachedTemporal(split, frames)
        except ValueError as e:
            print(f"cache {split}: skip ({e})")
            continue
        done, total = 0, len(ds)
        with torch.no_grad():
            for item in ds.items:
                dest = CACHE_DIR / item["key"]
                if dest.is_file():
                    done += 1
                    continue
                srs = []
                for k in range(frames):
                    sr = forward_native(backbone, item["lr"][k].unsqueeze(0).to(device))
                    srs.append(sr.squeeze(0).cpu())
                torch.save({"sr": torch.stack(srs), "lr": item["lr"],
                            "hr": item["hr"], "band_mask": item["band_mask"]}, dest)
                done += 1
        print(f"cache {split}: {done}/{total}")


def load_cached(split: str) -> list[dict]:
    return [{"blob": torch.load(p, weights_only=False), "key": p.stem}
            for p in sorted(CACHE_DIR.glob(f"{split}_*.pt"))]


def evaluate(head: TemporalFusionHead, rows: list[dict], device: torch.device,
             backbone_rows: list[dict] | None = None) -> dict[str, float]:
    loss_fn = SpectralSpatialLoss()
    head.eval()
    acc = {k: 0.0 for k in ("pixel", "spectral", "gradient", "source_consistency")}
    n = 0
    with torch.no_grad():
        for row in rows:
            b = row["blob"]
            sr = b["sr"].unsqueeze(0).to(device)
            lr = b["lr"].unsqueeze(0).to(device)
            hr = b["hr"].unsqueeze(0).to(device)
            mask = b["band_mask"].unsqueeze(0).to(device) if b["band_mask"].ndim == 1 else b["band_mask"].to(device)
            pred = head(sr, lr)
            comp = loss_fn.components(pred, hr, lr[:, 0], mask)
            for k in acc:
                acc[k] += comp[k].item()
            n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    backbone = SEN2SRModel(model_variant="lite", device=str(device), auto_download=False)
    ft = find_ft_checkpoint(REPO_ROOT)
    if ft is None:
        raise FileNotFoundError("Fine-tuned Lite weights missing.")
    load_checkpoint(ft, backbone)
    backbone.set_trainable(False)
    backbone.eval()
    print(f"backbone: FT weights from {ft.name} on {device}")

    build_cache(backbone, device, args.frames)
    head = TemporalFusionHead(frames=args.frames).to(device)
    out_path = REPO_ROOT / args.output
    if out_path.is_file() and not args.eval_only:
        head.load_state_dict(torch.load(out_path, map_location=device, weights_only=False)["head_state"])
        print(f"resumed head from {out_path.name}")

    val_rows = load_cached("val")
    test_rows = load_cached("test")
    if not args.eval_only:
        train_rows = load_cached("train")
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
        loss_fn = SpectralSpatialLoss()
        best = float("inf")
        for epoch in range(args.epochs):
            head.train()
            tot, nb = 0.0, 0
            perm = torch.randperm(len(train_rows))
            for bi in range(0, len(train_rows), args.batch_size):
                idx = perm[bi:bi + args.batch_size]
                sr = torch.stack([train_rows[i]["blob"]["sr"] for i in idx]).to(device)
                lr = torch.stack([train_rows[i]["blob"]["lr"] for i in idx]).to(device)
                hr = torch.stack([train_rows[i]["blob"]["hr"] for i in idx]).to(device)
                opt.zero_grad()
                pred = head(sr, lr)
                loss = loss_fn(pred, hr, lr[:, 0],
                               torch.stack([train_rows[i]["blob"]["band_mask"] for i in idx]).to(device))
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            sched.step()
            vm = evaluate(head, val_rows, device)
            vt = tot / max(nb, 1)
            print(f"epoch={epoch} train={vt:.6f} val_pixel={vm['pixel']:.6f} "
                  f"val_spec={vm['spectral']:.6f} val_grad={vm['gradient']:.6f} val_src={vm['source_consistency']:.6f}")
            if vm["pixel"] < best:
                best = vm["pixel"]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"head_state": head.state_dict(), "best_val": best,
                            "frames": args.frames}, out_path)
        print(f"best val pixel: {best:.6f} -> {out_path}")

    if out_path.is_file():
        head.load_state_dict(torch.load(out_path, map_location=device, weights_only=False)["head_state"])
    for name, rows in (("val", val_rows), ("test", test_rows)):
        vm = evaluate(head, rows, device)
        print(f"FUSED {name}: " + " ".join(f"{k}={v:.6f}" for k, v in vm.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
