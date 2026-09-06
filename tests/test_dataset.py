"""Dataset smoke tests — manifest integrity + PairedS2Dataset contracts."""

import csv
from pathlib import Path

import pytest
import torch

from ntro_srm.training.dataset import PairedS2Dataset

MANIFEST = Path(__file__).resolve().parents[1] / "datasets" / "paired" / "manifest.csv"
REQUIRED_COLUMNS = {"pair_id", "site_id", "indian_priority", "landcover", "task",
                    "split", "pair_type", "s2_path", "hr_path", "s2_date",
                    "hr_date", "dt_days", "tile_y", "tile_x", "lr_h", "lr_w"}

manifest_missing = not MANIFEST.is_file()
pytestmark = pytest.mark.skipif(manifest_missing, reason="paired manifest not built yet")


def _rows():
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("pair_id")]


def test_manifest_schema_and_splits():
    rows = _rows()
    assert rows, "manifest is empty"
    assert REQUIRED_COLUMNS.issubset(rows[0].keys())
    assert {r["pair_type"] for r in rows} <= {"wald_synthetic", "real_paired"}
    assert {"train", "val", "test"} <= {r["split"] for r in rows}
    assert any(r["indian_priority"] == "1" for r in rows), "no Indian-priority tiles"


def test_wald_sample_shapes():
    ds = PairedS2Dataset(MANIFEST, split="train")
    sample = next(ds[i] for i in range(len(ds))
                  if ds.rows[i]["pair_type"] == "wald_synthetic")
    assert sample.lr.shape[0] == 10 and sample.hr.shape[0] == 10
    assert sample.hr.shape[1] == sample.lr.shape[1] * 4
    assert sample.hr.shape[2] == sample.lr.shape[2] * 4
    assert sample.band_mask.sum().item() == 10.0
    assert torch.isfinite(sample.lr).all() and torch.isfinite(sample.hr).all()


def test_real_sample_common_band_mask():
    ds = PairedS2Dataset(MANIFEST)
    idx = next((i for i, r in enumerate(ds.rows) if r["pair_type"] == "real_paired"), None)
    if idx is None:
        pytest.skip("no real pairs in manifest")
    sample = ds[idx]
    assert tuple(sample.hr.shape) == (10, sample.lr.shape[1] * 4, sample.lr.shape[2] * 4)
    assert sample.band_mask.sum().item() >= 3.0  # at least RGB supervised
