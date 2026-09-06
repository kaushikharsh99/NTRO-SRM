"""Multi-temporal Wald dataset: K co-registered dates, one HR target.

For each Wald tile (site, y, x, 32px LR @40m -> 128px HR @10m), loads the same
window from K temporal chips (rank-0 = reference = HR target source) and
Wald-degrades each frame independently. All chips share the reference grid,
so tile coordinates transfer directly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ntro_srm.training.dataset import PairedS2Dataset
from ntro_srm.training.wald import wald_degrade


class TemporalS2Dataset(Dataset):
    """K-frame temporal samples aligned to paired-manifest Wald tiles."""

    def __init__(
        self,
        paired_manifest: str | Path,
        temporal_manifest: str | Path,
        split: str | None = None,
        frames: int = 3,
        normalize: bool = True,
        wald_scale: int = 4,
        bright_max: float = 0.35,
    ) -> None:
        self.paired = PairedS2Dataset(
            paired_manifest, split=split, normalize=False,
            pair_types=("wald_synthetic",))
        self.temporal_manifest = Path(temporal_manifest)
        if not self.temporal_manifest.is_file():
            raise FileNotFoundError(f"Temporal manifest not found: {self.temporal_manifest}")
        with self.temporal_manifest.open(newline="", encoding="utf-8") as fh:
            trows = [r for r in csv.DictReader(fh) if r.get("site_id")]
        by_site: dict[str, list[dict]] = {}
        for r in trows:
            by_site.setdefault(r["site_id"], []).append(r)
        self.root = self.temporal_manifest.parent.parent
        self.frames = frames
        self.normalize_flag = normalize
        self.wald_scale = wald_scale
        self.bright_max = bright_max
        self._cache: dict[str, torch.Tensor] = {}
        self.specs: list[tuple[str, int, int, list[str]]] = []
        for row, _, _ in self.paired.specs:
            site = row.get("site_id", "")
            dates = sorted(by_site.get(site, []), key=lambda r: float(r.get("cloud_pct") or 99))[:frames]
            if len(dates) < frames:
                continue
            y, x = int(row.get("tile_y", 0)), int(row.get("tile_x", 0))
            self.specs.append((site, y, x, [d["path"] for d in dates]))
        if not self.specs:
            raise ValueError("No temporal tiles: fetch chips first (scripts/fetch_temporal.py)")

    def __len__(self) -> int:
        return len(self.specs)

    def _load(self, rel: str) -> torch.Tensor:
        if rel not in self._cache:
            sample_ds = self.paired
            arr, _ = sample_ds._read(rel)
            t = torch.from_numpy(arr)
            if self.normalize_flag:
                from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a
                t = normalize_sentinel2_l2a(t, mode="auto", nodata_value=None)
            self._cache[rel] = t.float()
        return self._cache[rel]

    def __getitem__(self, idx: int) -> dict:
        site, y, x, paths = self.specs[idx]
        hrs, lrs = [], []
        for rel in paths:
            scene = self._load(rel)
            hr = scene[:, y:y + 128, x:x + 128]
            lr = wald_degrade(hr, scale=self.wald_scale)
            hrs.append(hr)
            lrs.append(lr)
        hr_ref = hrs[0]
        lr_stack = torch.stack(lrs)  # (K,10,32,32)
        # Brightness guard: clouds/snow break the temporal assumption.
        valid = bool((lr_stack[:, 2].mean() < self.bright_max).item())
        return {"lr": lr_stack, "hr": hr_ref,
                "band_mask": torch.ones(10),
                "meta": {"site_id": site, "y": y, "x": x}, "valid": valid}


def temporal_mean_lr(batch_lr: torch.Tensor) -> torch.Tensor:
    """Fuse by averaging LR frames before a single SR forward. (B,K,10,h,w)->(B,10,h,w)."""
    return batch_lr.mean(dim=1)
