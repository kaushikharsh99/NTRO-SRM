"""Tests for spatial, spectral, and geospatial evaluation."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from ntro_srm.evaluation import evaluate_arrays, evaluate_geotiffs
from ntro_srm.data.sentinel2 import S2_12BAND_ORDER
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


def _write_raster(
    path: Path,
    data: np.ndarray,
    transform: Affine,
    band_names: list[str] | None = None,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dataset:
        dataset.write(data.astype(np.float32))
        if band_names is not None:
            for index, name in enumerate(band_names, start=1):
                dataset.set_band_description(index, name)


def test_identical_arrays_have_ideal_scores() -> None:
    rng = np.random.default_rng(42)
    reference = rng.uniform(0.05, 0.8, size=(10, 32, 32))
    result = evaluate_arrays(reference, reference, band_names=S2_10BAND_NAMES)

    assert result["aggregate"]["mae"] == 0.0
    assert result["aggregate"]["rmse"] == 0.0
    assert result["aggregate"]["sam_degrees"] < 1e-6
    assert result["aggregate"]["mean_ssim"] > 0.999999
    assert result["aggregate"]["mean_psnr_db"] >= 119.0
    assert result["spectral_indices"]["ndvi"]["mae"] == 0.0


def test_bias_and_spectral_distortion_are_detected() -> None:
    reference = np.full((4, 16, 16), 0.2, dtype=np.float64)
    prediction = reference.copy()
    prediction[0] += 0.1
    result = evaluate_arrays(
        prediction,
        reference,
        band_names=["B02", "B03", "B04", "B08"],
    )

    assert np.isclose(result["per_band"]["B02"]["mae"], 0.1)
    assert result["aggregate"]["sam_degrees"] > 0.0
    assert result["aggregate"]["rmse"] > 0.0


def test_geotiff_source_consistency_and_reference_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform(0.1, 0.7, size=(10, 8, 8)).astype(np.float32)
    prediction = np.repeat(np.repeat(source, 4, axis=1), 4, axis=2)
    source_path = tmp_path / "source.tif"
    prediction_path = tmp_path / "prediction.tif"
    reference_path = tmp_path / "reference.tif"

    _write_raster(source_path, source, Affine(10, 0, 500000, 0, -10, 3000000), S2_10BAND_NAMES)
    _write_raster(prediction_path, prediction, Affine(2.5, 0, 500000, 0, -2.5, 3000000), S2_10BAND_NAMES)
    _write_raster(reference_path, prediction, Affine(2.5, 0, 500000, 0, -2.5, 3000000), S2_10BAND_NAMES)

    report = evaluate_geotiffs(
        prediction_path,
        source_path=source_path,
        reference_path=reference_path,
    )

    assert report.geospatial["source_crs_match"] is True
    assert report.geospatial["source_bounds_preserved"] is True
    assert np.isclose(report.geospatial["observed_scale_factor"], 4.0)
    assert report.source_consistency is not None
    assert report.source_consistency["aggregate"]["rmse"] < 1e-6
    assert report.reference_metrics is not None
    assert report.reference_metrics["aggregate"]["rmse"] < 1e-6

    output_path = report.write_json(tmp_path / "report.json")
    assert output_path.is_file()
    assert '"schema_version": "1.0"' in output_path.read_text(encoding="utf-8")


def test_untagged_standard_12_band_source_is_detected(tmp_path: Path) -> None:
    source = np.zeros((12, 4, 4), dtype=np.float32)
    for index in range(12):
        source[index] = 0.1 + index * 0.01
    selected = source[[S2_12BAND_ORDER.index(name) for name in S2_10BAND_NAMES]]
    prediction = np.repeat(np.repeat(selected, 4, axis=1), 4, axis=2)
    source_path = tmp_path / "source_12_band.tif"
    prediction_path = tmp_path / "prediction_10_band.tif"

    _write_raster(source_path, source, Affine(10, 0, 500000, 0, -10, 3000000))
    _write_raster(
        prediction_path,
        prediction,
        Affine(2.5, 0, 500000, 0, -2.5, 3000000),
        S2_10BAND_NAMES,
    )

    report = evaluate_geotiffs(prediction_path, source_path=source_path)
    assert report.source is not None
    assert report.source.band_names == S2_12BAND_ORDER
    assert report.source_consistency is not None
    assert report.source_consistency["aggregate"]["rmse"] < 1e-6
