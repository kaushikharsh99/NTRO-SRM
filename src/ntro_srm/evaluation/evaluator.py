"""GeoTIFF alignment and report generation for SR evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch

from ntro_srm.evaluation.metrics import evaluate_arrays
from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a
from ntro_srm.data.sentinel2 import S2_12BAND_ORDER
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


@dataclass(frozen=True)
class RasterSummary:
    path: str
    crs: str | None
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    shape: tuple[int, int, int]
    band_names: list[str]


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    generated_at: str
    prediction: RasterSummary
    source: RasterSummary | None
    reference: RasterSummary | None
    geospatial: dict
    source_consistency: dict | None
    reference_metrics: dict | None
    limitations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination


def _band_names(dataset: rasterio.DatasetReader) -> list[str]:
    tags = dataset.tags()
    names: list[str] = []
    known_names = ["B01", *S2_10BAND_NAMES, "B09", "B10"]
    for index in range(1, dataset.count + 1):
        description = (dataset.descriptions[index - 1] or "").upper()
        tagged = tags.get(f"BAND_{index}", "").upper()
        detected = next((band for band in known_names if band in tagged or band in description), None)
        names.append(detected or f"band_{index}")

    if all(name.startswith("band_") for name in names):
        if dataset.count == 13:
            return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
        if dataset.count == 12:
            return list(S2_12BAND_ORDER)
        if dataset.count == 10:
            return list(S2_10BAND_NAMES)
        if dataset.count == 4:
            return ["B02", "B03", "B04", "B08"]
        if dataset.count == 3:
            return ["B04", "B03", "B02"]
    return names


def _summary(path: Path) -> RasterSummary:
    if not path.is_file():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")
    with rasterio.open(path) as dataset:
        return RasterSummary(
            path=str(path.resolve()),
            crs=dataset.crs.to_string() if dataset.crs else None,
            bounds=tuple(float(value) for value in dataset.bounds),
            resolution=tuple(float(abs(value)) for value in dataset.res),
            shape=(dataset.count, dataset.height, dataset.width),
            band_names=_band_names(dataset),
        )


def _normalise(data: np.ndarray) -> np.ndarray:
    return normalize_sentinel2_l2a(
        torch.from_numpy(data), mode="auto", nodata_value=None
    ).numpy().astype(np.float64)


def _read_selected_on_grid(
    path: Path,
    selected_bands: Sequence[str],
    *,
    target_crs,
    target_transform,
    target_shape: tuple[int, int],
    resampling: Resampling,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as dataset:
        if dataset.crs is None or target_crs is None:
            raise ValueError(f"Both source and target rasters must define a CRS: {path}")
        names = _band_names(dataset)
        missing = [name for name in selected_bands if name not in names]
        if missing:
            raise ValueError(f"{path} does not contain required bands: {missing}")

        output = np.empty((len(selected_bands), *target_shape), dtype=np.float32)
        valid = np.ones(target_shape, dtype=bool)
        for output_index, band_name in enumerate(selected_bands):
            source_index = names.index(band_name) + 1
            source = dataset.read(source_index)
            destination = np.full(target_shape, np.nan, dtype=np.float32)
            reproject(
                source=source,
                destination=destination,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                src_nodata=dataset.nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
            output[output_index] = destination
            valid &= np.isfinite(destination)
        return _normalise(output), valid


def _common_bands(first: RasterSummary, second: RasterSummary) -> list[str]:
    common = [name for name in first.band_names if name in second.band_names and name.startswith("B")]
    if not common:
        if first.shape[0] != second.shape[0]:
            raise ValueError("No band metadata matches and raster band counts differ")
        return list(first.band_names)
    return common


def _bounds_close(first: RasterSummary, second: RasterSummary) -> bool:
    tolerance = max(*first.resolution, *second.resolution) * 1.1
    return bool(np.allclose(first.bounds, second.bounds, atol=tolerance, rtol=0.0))


def evaluate_geotiffs(
    prediction_path: str | Path,
    *,
    source_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    scale_factor: float = 4.0,
) -> EvaluationReport:
    """Evaluate an SR GeoTIFF against its LR source and/or an HR reference.

    Reference imagery is bilinearly aligned to the prediction grid. For source
    consistency, the prediction is area-averaged back to the source grid before
    metrics are calculated.
    """
    prediction_file = Path(prediction_path)
    prediction_summary = _summary(prediction_file)
    if source_path is None and reference_path is None:
        raise ValueError("Provide source_path, reference_path, or both")

    source_file = Path(source_path) if source_path is not None else None
    reference_file = Path(reference_path) if reference_path is not None else None
    source_summary = _summary(source_file) if source_file is not None else None
    reference_summary = _summary(reference_file) if reference_file is not None else None
    limitations: list[str] = []

    geospatial: dict = {
        "prediction_has_crs": prediction_summary.crs is not None,
        "source_crs_match": None,
        "source_bounds_preserved": None,
        "reference_crs_match": None,
        "reference_bounds_match": None,
        "observed_scale_factor": None,
    }

    source_metrics = None
    if source_file is not None and source_summary is not None:
        common = _common_bands(prediction_summary, source_summary)
        with rasterio.open(source_file) as source_dataset:
            grid = {
                "target_crs": source_dataset.crs,
                "target_transform": source_dataset.transform,
                "target_shape": (source_dataset.height, source_dataset.width),
            }
        source, source_valid = _read_selected_on_grid(
            source_file, common, resampling=Resampling.bilinear, **grid
        )
        downsampled, prediction_valid = _read_selected_on_grid(
            prediction_file, common, resampling=Resampling.average, **grid
        )
        source_metrics = evaluate_arrays(
            downsampled,
            source,
            band_names=common,
            valid_mask=source_valid & prediction_valid,
            scale_factor=1.0,
        )
        geospatial["source_crs_match"] = prediction_summary.crs == source_summary.crs
        geospatial["source_bounds_preserved"] = _bounds_close(prediction_summary, source_summary)
        if prediction_summary.resolution[0] > 0:
            geospatial["observed_scale_factor"] = float(
                source_summary.resolution[0] / prediction_summary.resolution[0]
            )
        limitations.append(
            "Source-consistency metrics only test whether downsampled SR agrees with the observation; "
            "they do not prove that reconstructed high-frequency detail is correct."
        )

    reference_metrics = None
    if reference_file is not None and reference_summary is not None:
        common = _common_bands(prediction_summary, reference_summary)
        with rasterio.open(prediction_file) as prediction_dataset:
            grid = {
                "target_crs": prediction_dataset.crs,
                "target_transform": prediction_dataset.transform,
                "target_shape": (prediction_dataset.height, prediction_dataset.width),
            }
        prediction, prediction_valid = _read_selected_on_grid(
            prediction_file, common, resampling=Resampling.bilinear, **grid
        )
        reference, reference_valid = _read_selected_on_grid(
            reference_file, common, resampling=Resampling.bilinear, **grid
        )
        reference_metrics = evaluate_arrays(
            prediction,
            reference,
            band_names=common,
            valid_mask=prediction_valid & reference_valid,
            scale_factor=scale_factor,
        )
        geospatial["reference_crs_match"] = prediction_summary.crs == reference_summary.crs
        geospatial["reference_bounds_match"] = _bounds_close(prediction_summary, reference_summary)

    return EvaluationReport(
        schema_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        prediction=prediction_summary,
        source=source_summary,
        reference=reference_summary,
        geospatial=geospatial,
        source_consistency=source_metrics,
        reference_metrics=reference_metrics,
        limitations=limitations,
    )
