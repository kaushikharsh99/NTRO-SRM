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
    ) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")
        with self.manifest.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("pair_id")]
        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        if not rows:
            raise ValueError(f"No pairs in {self.manifest} for split={split!r}")
        self.rows = rows
        self.root = self.manifest.parent
        self.normalize = normalize
        self.wald_scale = wald_scale

    def __len__(self) -> int:
        return len(self.rows)

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

    def __getitem__(self, idx: int) -> PairedSample:
        row = self.rows[idx]
        pair_type = row.get("pair_type", "wald_synthetic")
        s2_arr, _ = self._read(row["s2_path"])
        s2 = torch.from_numpy(s2_arr)
        if self.normalize:
            s2 = normalize_sentinel2_l2a(s2, mode="auto", nodata_value=None)

        y, x = int(row.get("tile_y", 0)), int(row.get("tile_x", 0))
        lr_h, lr_w = int(row.get("lr_h", 32)), int(row.get("lr_w", 32))
        if pair_type == "wald_synthetic":
            # HR target = real S2 window at 10m; LR source = Wald 40m.
            hr = s2[:, y:y + lr_h * 4, x:x + lr_w * 4]
            lr = wald_degrade(hr, scale=self.wald_scale)
            mask = torch.ones(10)
        else:
            hr_arr, _ = self._read(row["hr_path"])
            hr = torch.from_numpy(hr_arr)
            if self.normalize:
                # HR refs are typically uint8 RGBN or float; scale heuristically.
                hr = normalize_sentinel2_l2a(hr, mode="auto", nodata_value=None)
            lr = s2[:, y:y + lr_h, x:x + lr_w]
            # Supervise common RGBN bands only; keep 10-band shape with mask.
            mask = torch.zeros(10)
            common = (row.get("common_bands", "B02,B03,B04,B08") or "").split(",")
            hr_full = torch.zeros_like(lr.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1))
            for band in [b.strip() for b in common if b.strip() in _S2_INDEX]:
                s2_i = _S2_INDEX[band]
                if band in _RGBN_INDEX and _RGBN_INDEX[band] < hr.shape[0]:
                    hr_full[s2_i] = hr[_RGBN_INDEX[band]]
                    mask[s2_i] = 1.0
            hr = hr_full
        return PairedSample(lr=lr.float(), hr=hr.float(), band_mask=mask,
                            meta={"pair_id": row["pair_id"], "pair_type": pair_type,
                                  "site_id": row.get("site_id", ""),
                                  "split": row.get("split", "")})
