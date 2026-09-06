"""Self-contained tests for the paired-data manifest and dataset contracts."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import torch

from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.training.dataset import PairedS2Dataset

PROJECT_MANIFEST = Path(__file__).resolve().parents[1] / "datasets" / "paired" / "manifest.csv"
REQUIRED_COLUMNS = {
    "pair_id", "site_id", "indian_priority", "landcover", "task", "split",
    "pair_type", "s2_path", "hr_path", "s2_date", "hr_date", "dt_days",
    "tile_y", "tile_x", "lr_h", "lr_w",
}


def _write_raster(path: Path, data: np.ndarray, band_names: list[str]) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=data.shape[0],
        dtype="float32",
        crs=CRS.from_epsg(32643),
        transform=Affine(10, 0, 500000, 0, -10, 3000000),
    ) as dataset:
        dataset.write(data.astype(np.float32))
        for index, band_name in enumerate(band_names, start=1):
            dataset.set_band_description(index, band_name)


def _write_manifest(path: Path, row: dict[str, object]) -> None:
    columns = [
        "pair_id", "site_id", "indian_priority", "landcover", "task", "split",
        "pair_type", "s2_path", "hr_path", "s2_date", "hr_date", "dt_days",
        "cloud_pct", "crs", "align_rmse_px", "tile_y", "tile_x", "lr_h",
        "lr_w", "common_bands", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def test_project_manifest_schema_and_splits() -> None:
    if not PROJECT_MANIFEST.is_file():
        pytest.skip("paired manifest not built yet")
    with PROJECT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("pair_id")]

    assert rows
    assert REQUIRED_COLUMNS.issubset(rows[0].keys())
    assert {row["pair_type"] for row in rows} <= {"wald_synthetic", "real_paired"}
    assert {"train", "val", "test"} <= {row["split"] for row in rows}
    assert any(row["indian_priority"] == "1" for row in rows)


def test_wald_sample_shapes_without_external_data(tmp_path: Path) -> None:
    source_path = tmp_path / "source.tif"
    manifest_path = tmp_path / "manifest.csv"
    source = np.random.default_rng(4).uniform(0.05, 0.8, (10, 128, 128))
    _write_raster(source_path, source, S2_10BAND_NAMES)
    _write_manifest(
        manifest_path,
        {
            "pair_id": "wald-1", "site_id": "site-a", "split": "train",
            "pair_type": "wald_synthetic", "s2_path": source_path,
            "tile_y": 0, "tile_x": 0, "lr_h": 32, "lr_w": 32,
        },
    )

    sample = PairedS2Dataset(manifest_path, split="train")[0]
    assert sample.lr.shape == (10, 32, 32)
    assert sample.hr.shape == (10, 128, 128)
    assert sample.band_mask.sum().item() == 10.0
    assert torch.isfinite(sample.lr).all() and torch.isfinite(sample.hr).all()


def test_real_sample_supervises_common_bands_without_external_data(tmp_path: Path) -> None:
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    manifest_path = tmp_path / "manifest.csv"
    rng = np.random.default_rng(8)
    _write_raster(source_path, rng.uniform(0.05, 0.8, (10, 32, 32)), S2_10BAND_NAMES)
    _write_raster(
        reference_path,
        rng.uniform(0.05, 0.8, (4, 128, 128)),
        ["B04", "B03", "B02", "B08"],
    )
    _write_manifest(
        manifest_path,
        {
            "pair_id": "real-1", "site_id": "site-b", "split": "val",
            "pair_type": "real_paired", "s2_path": source_path,
            "hr_path": reference_path, "tile_y": 0, "tile_x": 0,
            "lr_h": 32, "lr_w": 32, "common_bands": "B02,B03,B04,B08",
        },
    )

    sample = PairedS2Dataset(manifest_path, split="val")[0]
    assert sample.lr.shape == (10, 32, 32)
    assert sample.hr.shape == (10, 128, 128)
    assert sample.band_mask.sum().item() == 4.0
    assert torch.isfinite(sample.lr).all() and torch.isfinite(sample.hr).all()
