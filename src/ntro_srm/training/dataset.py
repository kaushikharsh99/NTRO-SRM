"""Paired Sentinel-2 dataset for training / fine-tuning.

Supports two pair types (see manifest `pair_type`):
- wald_synthetic: HR = real S2 10m tile, LR = Wald-degraded 40m tile.
  Full 10-band supervision. Works everywhere incl. India with no HR cost.
- real_paired: LR = S2 10m tile, HR = co-registered HR reference (NAIP /
  Maxar / OpenSR) resampled to 2.5m. Supervision mask covers common bands
  only (typically RGBN); red-edge/SWIR fall back to source-consistency.

Manifest schema (datasets/paired/manifest.csv):
pair_id,site_id,indian_priority,landcover,task,split,pair_type,
s2_path,hr_path,s2_date,hr_date,dt_days,cloud_pct,crs,
align_rmse_px,tile_y,tile_x,lr_h,lr_w,notes
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.training.wald import wald_degrade

_RGBN_INDEX = {"B04": 0, "B03": 1, "B02": 2, "B08": 3}
_S2_INDEX = {name: i for i, name in enumerate(S2_10BAND_NAMES)}


@dataclass
class PairedSample:
    lr: torch.Tensor        # (10,H_lr,W_lr) normalized
    hr: torch.Tensor        # (10,H_hr,W_hr) normalized (NaN where unsupervised)
    band_mask: torch.Tensor  # (10,) 1.0 where HR supervises, 0.0 otherwise
    meta: dict


class PairedS2Dataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str | None = None,
        normalize: bool = True,
        wald_scale: int = 4,
        pair_types: list[str] | tuple[str, ...] | None = None,
        tile_lr: int = 32,
        tile_stride: int = 32,
        augment: bool = False,
    ) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")
        with self.manifest.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("pair_id")]
        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        if pair_types is not None:
            wanted = set(pair_types)
            rows = [r for r in rows if r.get("pair_type", "wald_synthetic") in wanted]
        if not rows:
            raise ValueError(f"No pairs in {self.manifest} for split={split!r}")
        self.root = self.manifest.parent
        self.normalize = normalize
        self.wald_scale = wald_scale
        self.tile_lr = tile_lr
        self.tile_stride = tile_stride
        self.augment = augment
        self._cache: dict[str, torch.Tensor] = {}
        # Expand rows into fixed-size tile specs. Wald rows are already
        # single tiles; full-scene real rows are split into a tile grid so
        # every batch stacks to identical shapes.
        self.specs: list[tuple[dict, int, int]] = []
        for row in rows:
            if row.get("pair_type", "wald_synthetic") == "wald_synthetic":
                self.specs.append((row, 0, 0))
            else:
                h = int(row.get("lr_h", 0)) or 0
                w = int(row.get("lr_w", 0)) or 0
                self.specs.extend((row, ty, tx) for ty, tx in self._tile_grid(h, w))
        if not self.specs:
            raise ValueError(f"No tiles in {self.manifest} for split={split!r}")

    def _tile_grid(self, h: int, w: int) -> list[tuple[int, int]]:
        t, s = self.tile_lr, self.tile_stride
        if h <= 0 or w <= 0:
            return [(0, 0)]
        ys = list(range(0, max(1, h - t + 1), s))
        xs = list(range(0, max(1, w - t + 1), s))
        if ys[-1] + t < h:
            ys.append(h - t)
        if xs[-1] + t < w:
            xs.append(w - t)
        return [(y, x) for y in ys for x in xs]

    def __len__(self) -> int:
        return len(self.specs)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        candidates = [Path.cwd() / p, self.root / p]
        candidates += [self.root.parent / p, self.root.parent.parent / p]
        for c in candidates:
            if c.is_file():
                return c
        return (Path.cwd() / p).resolve()

    def _read(self, path: str) -> tuple[np.ndarray, dict]:
        p = self._resolve(path)
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
            meta = {"crs": str(src.crs), "transform": src.transform,
                    "descriptions": list(src.descriptions or [])}
        return arr, meta

    def _cached_s2(self, path: str) -> torch.Tensor:
        if path not in self._cache:
            arr, _ = self._read(path)
            s2 = torch.from_numpy(arr)
            if self.normalize:
                s2 = normalize_sentinel2_l2a(s2, mode="auto", nodata_value=None)
            self._cache[path] = s2.float()
        return self._cache[path]

    def __getitem__(self, idx: int) -> PairedSample:
        row, dty, dtx = self.specs[idx]
        pair_type = row.get("pair_type", "wald_synthetic")
        t = self.tile_lr
        if pair_type == "wald_synthetic":
            s2 = self._cached_s2(row["s2_path"])
            y, x = int(row.get("tile_y", 0)), int(row.get("tile_x", 0))
            lr_h, lr_w = int(row.get("lr_h", 32)), int(row.get("lr_w", 32))
            # HR target = real S2 window at 10m; LR source = Wald 40m.
            hr = s2[:, y:y + lr_h * 4, x:x + lr_w * 4]
            lr = wald_degrade(hr, scale=self.wald_scale)
            mask = torch.ones(10)
            meta = {"pair_id": row["pair_id"], "pair_type": pair_type,
                    "site_id": row.get("site_id", ""), "split": row.get("split", "")}
        else:
            s2 = self._cached_s2(row["s2_path"])
            y, x = int(row.get("tile_y", 0)), int(row.get("tile_x", 0))
            lr_h, lr_w = int(row.get("lr_h", 32)), int(row.get("lr_w", 32))
            key = f"hr::{row['hr_path']}"
            if key not in self._cache:
                hr_arr, _ = self._read(row["hr_path"])
                hr_full = torch.from_numpy(hr_arr)
                if self.normalize:
                    # HR refs are typically uint8 RGBN or float; scale heuristically.
                    hr_full = normalize_sentinel2_l2a(hr_full, mode="auto", nodata_value=None)
                self._cache[key] = hr_full.float()
            hr_scene = self._cache[key]
            lr_window = s2[:, y:y + lr_h, x:x + lr_w]
            # Align HR to 4x the LR window (refs are pre-resampled to 2.5m).
            expect_h, expect_w = lr_window.shape[-2] * 4, lr_window.shape[-1] * 4
            if tuple(hr_scene.shape[-2:]) != (expect_h, expect_w):
                hr_scene = torch.nn.functional.interpolate(
                    hr_scene.unsqueeze(0), size=(expect_h, expect_w),
                    mode="bilinear", align_corners=False).squeeze(0)
            lr = lr_window[:, dty:dty + t, dtx:dtx + t]
            hr_crop = hr_scene[:, dty * 4:(dty + t) * 4, dtx * 4:(dtx + t) * 4]
            # Supervise common RGBN bands only; keep 10-band shape with mask.
            mask = torch.zeros(10)
            common = (row.get("common_bands", "B02,B03,B04,B08") or "").split(",")
            hr = torch.zeros(10, t * 4, t * 4)
            for band in [b.strip() for b in common if b.strip() in _S2_INDEX]:
                s2_i = _S2_INDEX[band]
                if band in _RGBN_INDEX and _RGBN_INDEX[band] < hr_crop.shape[0]:
                    hr[s2_i] = hr_crop[_RGBN_INDEX[band]]
                    mask[s2_i] = 1.0
            # Cross-sensor radiometry: NAIP DNs (0-255) are not S2 reflectance.
            # Match each supervised band's mean/std to the LR tile so the model
            # learns texture/detail instead of a global brightness bias.
            for s2_i in range(10):
                if mask[s2_i].item() > 0:
                    ref, obs = hr[s2_i], lr[s2_i]
                    std = ref.std() + 1e-6
                    hr[s2_i] = (ref - ref.mean()) / std * (obs.std() + 1e-6) + obs.mean()
            meta = {"pair_id": f"{row['pair_id']}@{dty},{dtx}", "pair_type": pair_type,
                    "site_id": row.get("site_id", ""), "split": row.get("split", "")}
        if self.augment:
            lr, hr = self._augment_pair(lr, hr)
        return PairedSample(lr=lr.float(), hr=hr.float(), band_mask=mask,
                            meta=meta)

    @staticmethod
    def _augment_pair(lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply identical random flip/rot90 to an LR/HR tile pair."""
        if torch.rand(()) < 0.5:
            lr, hr = torch.flip(lr, [-1]), torch.flip(hr, [-1])
        if torch.rand(()) < 0.5:
            lr, hr = torch.flip(lr, [-2]), torch.flip(hr, [-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            lr, hr = torch.rot90(lr, k, [-2, -1]), torch.rot90(hr, k, [-2, -1])
        return lr, hr
