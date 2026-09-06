"""Scientific quality-assessment and thematic-product service for NTRO-SRM web jobs.

Wraps the :mod:`ntro_srm.evaluation` package so that every super-resolution job
produced by the web application ships with:

* **Accuracy assessment** — Wald's synthesis protocol (10 m -> 40 m -> 10 m) scored
  against the observed Sentinel-2 image, alongside a bicubic baseline so the gain
  attributable to the network is explicit.
* **Radiometric consistency** — Wald's consistency property, verifying that the
  2.5 m product collapses back onto the observed 10 m reflectance.
* **Uncertainty** — a per-pixel confidence field separating observed structure from
  detail the network inferred, as required by the problem statement.
* **Application products** — spectral indices (crop, water, built-up, burn) computed
  on the super-resolved grid, with statistics and a sharpening gain versus the
  native-resolution index.

Heavy raster previews are rendered lazily: only the headline layers are written when
the job completes, and any other layer is materialised on first request and cached in
the job directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.warp import transform as warp_transform

from ntro_srm.evaluation import (
    INDEX_REGISTRY,
    METRIC_META,
    apply_colormap_rgba,
    build_report,
    compute_index,
    consistency_check,
    estimate_uncertainty,
    index_delta_statistics,
    index_statistics,
    registry_as_json,
    render_index_png,
    render_uncertainty_png,
    save_report,
    spectral_fidelity,
    wald_protocol_validate,
)
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES

# Sentinel-2 MSI central wavelengths (nm) in canonical 10-band order.
S2_WAVELENGTHS_NM: list[float] = [
    492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4
]

# Three-band composites offered as map layers: (name, band indices R/G/B, human label).
COMPOSITES: dict[str, dict[str, Any]] = {
    "rgb": {
        "label": "Natural colour",
        "bands": (2, 1, 0),
        "detail": "B04 / B03 / B02 — true colour",
    },
    "cir": {
        "label": "Colour infrared",
        "bands": (6, 2, 1),
        "detail": "B08 / B04 / B03 — vegetation appears red",
    },
    "swir": {
        "label": "SWIR composite",
        "bands": (9, 6, 2),
        "detail": "B12 / B08 / B04 — burn scars, geology, moisture",
    },
    "agri": {
        "label": "Agriculture",
        "bands": (8, 6, 1),
        "detail": "B11 / B08 / B03 — crop vigour and field boundaries",
    },
    "geology": {
        "label": "Geology",
        "bands": (9, 8, 3),
        "detail": "B12 / B11 / B05 — lithology and bare surfaces",
    },
}

# Analysis rasters derived from the uncertainty estimate.
ANALYSIS_LAYERS: dict[str, dict[str, str]] = {
    "confidence": {
        "label": "Reconstruction confidence",
        "cmap": "rdylgn",
        "detail": "Green = detail supported by the observation; red = largely model-inferred",
    },
    "novelty": {
        "label": "Synthesised detail",
        "cmap": "inferno",
        "detail": "Magnitude of structure the network added beyond bicubic interpolation",
    },
    "spread": {
        "label": "Ensemble spread",
        "cmap": "inferno",
        "detail": "Per-pixel standard deviation across the test-time-augmentation ensemble",
    },
}

# Layers rendered eagerly when a job finishes; everything else is lazy.
EAGER_INDEX_LAYERS: tuple[str, ...] = ("ndvi", "ndwi")

_SOURCES: tuple[str, ...] = ("lr", "sr", "bicubic")


def available_layer_names() -> list[str]:
    """Enumerate every preview layer the API is willing to serve.

    Returns
    -------
    list[str]
        Layer identifiers of the form ``<source>_<product>`` where source is one of
        ``lr`` / ``sr`` / ``bicubic`` and product is a composite key, a spectral index
        key, or (for ``sr`` only) an analysis-layer key.
    """
    names: list[str] = []
    for source in _SOURCES:
        for key in COMPOSITES:
            names.append(f"{source}_{key}")
        for key in INDEX_REGISTRY:
            names.append(f"{source}_{key}")
    for key in ANALYSIS_LAYERS:
        names.append(f"sr_{key}")
    return names


def parse_layer_name(layer_name: str) -> tuple[str, str, str]:
    """Split a layer identifier into ``(source, product, kind)``.

    Parameters
    ----------
    layer_name : str
        Identifier such as ``sr_rgb``, ``lr_ndvi`` or ``sr_confidence``.

    Returns
    -------
    tuple[str, str, str]
        ``(source, product, kind)`` where kind is ``"composite"``, ``"index"`` or
        ``"analysis"``.

    Raises
    ------
    ValueError
        If the identifier does not name a servable layer.
    """
    for key in ANALYSIS_LAYERS:
        if layer_name == f"sr_{key}":
            return ("sr", key, "analysis")

    source, _, product = layer_name.partition("_")
    if source not in _SOURCES or not product:
        raise ValueError(f"Unknown preview layer '{layer_name}'.")
    if product in COMPOSITES:
        return (source, product, "composite")
    if product in INDEX_REGISTRY:
        return (source, product, "index")
    raise ValueError(f"Unknown preview layer '{layer_name}'.")


def layer_catalog() -> dict[str, Any]:
    """Describe every layer group for the web UI's layer picker.

    Returns
    -------
    dict
        ``{"composites": [...], "indices": [...], "analysis": [...]}`` where each entry
        carries the key, human label, explanatory detail and (for indices) the colour
        legend and interpretation classes.
    """
    return {
        "composites": [
            {"key": k, "label": v["label"], "detail": v["detail"], "bands": list(v["bands"])}
            for k, v in COMPOSITES.items()
        ],
        "indices": registry_as_json(),
        "analysis": [
            {"key": k, "label": v["label"], "detail": v["detail"], "cmap": v["cmap"]}
            for k, v in ANALYSIS_LAYERS.items()
        ],
        "wavelengths_nm": S2_WAVELENGTHS_NM,
        "band_names": list(S2_10BAND_NAMES),
        "metric_meta": METRIC_META,
    }


@dataclass
class AnalysisArtifacts:
    """Outputs of a completed quality-assessment pass.

    Attributes
    ----------
    payload : dict
        JSON-serialisable analysis block attached to the job result.
    confidence_array : numpy.ndarray or None
        Per-pixel reconstruction confidence on the super-resolved grid.
    """

    payload: dict[str, Any]
    confidence_array: Optional[np.ndarray]


class AnalysisService:
    """Runs quality assessment for a job and renders its raster products on demand."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------
    def run(
        self,
        job_id: str,
        job_dir: Path,
        lr_tensor: torch.Tensor,
        sr_tensor: torch.Tensor,
        predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scene_meta: Optional[dict[str, Any]] = None,
        run_wald: bool = True,
        uncertainty_members: int = 0,
        index_keys: Optional[list[str]] = None,
        crs: Any = None,
        sr_transform: Any = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> AnalysisArtifacts:
        """Execute the full assessment and write the QA report plus headline rasters.

        Parameters
        ----------
        job_id : str
            Identifier of the job being assessed.
        job_dir : Path
            Directory holding the job's products.
        lr_tensor : torch.Tensor
            Observed 10 m reflectance, shape ``(10, H, W)`` in ``[0, 1]``.
        sr_tensor : torch.Tensor
            Super-resolved 2.5 m reflectance, shape ``(10, 4H, 4W)``.
        predict_fn : callable, optional
            ``(C, h, w) -> (C, 4h, 4w)`` forward pass. Required for Wald validation and
            for ensemble uncertainty; when absent both degrade gracefully.
        scene_meta : dict, optional
            Free-form job metadata embedded in the report.
        run_wald : bool, default=True
            Whether to run the synthesis protocol (costs one forward pass at 1/16 area).
        uncertainty_members : int, default=0
            Test-time-augmentation ensemble size. ``0`` uses the free novelty-only
            estimator and performs no extra forward passes.
        index_keys : list[str], optional
            Spectral indices to summarise. Defaults to every registered index.
        crs, sr_transform : optional
            Georeferencing used when exporting the confidence raster.
        progress_callback : callable, optional
            ``(message, percent)`` sink for coarse progress reporting.

        Returns
        -------
        AnalysisArtifacts
            The JSON payload for the job result and the confidence array.
        """
        started = time.perf_counter()
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        band_names = list(S2_10BAND_NAMES)

        def _tick(message: str, percent: int) -> None:
            if progress_callback:
                try:
                    progress_callback(message, percent)
                except Exception:  # progress reporting must never break the job
                    pass

        warnings: list[str] = []

        # 1. Radiometric consistency — free, always runs.
        _tick("Verifying radiometric consistency against the observation...", 82)
        consistency = None
        fidelity = None
        try:
            consistency = consistency_check(lr_tensor, sr_tensor, factor=4, band_names=band_names)
            fidelity = spectral_fidelity(lr_tensor, sr_tensor, factor=4, band_names=band_names)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Consistency check unavailable: {exc}")

        # 2. Wald synthesis protocol — one forward pass at quarter scale.
        wald = None
        if run_wald and predict_fn is not None:
            _tick("Running Wald synthesis validation (10 m -> 40 m -> 10 m)...", 85)
            try:
                wald = wald_protocol_validate(lr_tensor, predict_fn, factor=4, band_names=band_names)
            except Exception as exc:
                warnings.append(f"Wald validation could not be completed: {exc}")
        elif run_wald:
            warnings.append("Wald validation skipped: no forward pass available for this job.")

        # 3. Uncertainty.
        _tick("Estimating reconstruction uncertainty...", 88)
        uncertainty = None
        try:
            members = max(0, int(uncertainty_members))
            uncertainty = estimate_uncertainty(
                lr_tensor,
                sr_tensor,
                predict_fn=predict_fn if members >= 2 else None,
                n_ensemble=members if members >= 2 else 1,
                factor=4,
                progress_callback=lambda m, p: _tick(m, 88 + int(p * 0.04)),
            )
        except Exception as exc:
            warnings.append(f"Uncertainty estimation failed: {exc}")

        # 4. Thematic products.
        _tick("Deriving thematic products at 2.5 m...", 92)
        keys = list(index_keys) if index_keys else list(INDEX_REGISTRY.keys())
        sr_np = _as_numpy(sr_tensor)
        lr_np = _as_numpy(lr_tensor)
        index_stats: list[dict[str, Any]] = []
        for key in keys:
            try:
                sr_arr = compute_index(sr_np, key, band_names=band_names)
                stats = index_statistics(sr_arr, key)
                lr_arr = compute_index(lr_np, key, band_names=band_names)
                stats["delta"] = index_delta_statistics(lr_arr, sr_arr, key)
                index_stats.append(stats)
                if key in EAGER_INDEX_LAYERS:
                    render_index_png(sr_arr, key, job_dir / f"sr_{key}.png")
                    render_index_png(lr_arr, key, job_dir / f"lr_{key}.png")
            except Exception as exc:
                warnings.append(f"Index '{key}' could not be computed: {exc}")

        # 5. Uncertainty rasters and the georeferenced confidence product.
        confidence_array: Optional[np.ndarray] = None
        if uncertainty is not None:
            confidence_array = np.asarray(uncertainty.confidence_map, dtype=np.float32)
            try:
                render_uncertainty_png(uncertainty, job_dir / "sr_confidence.png", "confidence")
                render_uncertainty_png(uncertainty, job_dir / "sr_novelty.png", "novelty")
                if uncertainty.n_ensemble > 1:
                    render_uncertainty_png(uncertainty, job_dir / "sr_spread.png", "std")
            except Exception as exc:
                warnings.append(f"Uncertainty preview rendering failed: {exc}")
            if crs is not None and sr_transform is not None:
                try:
                    _write_single_band_geotiff(
                        job_dir / f"{job_id}_confidence_2.5m.tif",
                        confidence_array,
                        sr_transform,
                        crs,
                        description="NTRO-SRM reconstruction confidence [0-1]",
                    )
                except Exception as exc:
                    warnings.append(f"Confidence GeoTIFF export failed: {exc}")

        # 6. QA report.
        _tick("Compiling quality assessment report...", 94)
        report_paths: dict[str, str] = {}
        summary: dict[str, Any] = {}
        try:
            report = build_report(
                job_id,
                scene_meta or {},
                wald=wald,
                consistency=consistency,
                uncertainty=uncertainty,
                spectral_fidelity=fidelity,
                indices=index_stats,
            )
            report_paths = save_report(report, job_dir)
            summary = report.summary
        except Exception as exc:
            warnings.append(f"Quality report generation failed: {exc}")

        payload: dict[str, Any] = {
            "summary": summary,
            "wald": _to_dict(wald),
            "consistency": _to_dict(consistency),
            "uncertainty": _to_dict(uncertainty),
            "spectral_fidelity": fidelity,
            "indices": index_stats,
            "warnings": warnings,
            "report": {
                "json": f"/api/sr/jobs/{job_id}/download/report",
                "markdown": f"/api/sr/jobs/{job_id}/download/report-md",
            },
            "confidence_geotiff": (
                f"/api/sr/jobs/{job_id}/download/confidence" if confidence_array is not None else None
            ),
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
        return AnalysisArtifacts(payload=payload, confidence_array=confidence_array)

    # ------------------------------------------------------------------
    # Lazy raster rendering
    # ------------------------------------------------------------------
    def render_layer(
        self,
        job_dir: Path,
        layer_name: str,
        lr_path: Path,
        sr_path: Path,
        confidence_path: Optional[Path] = None,
    ) -> Path:
        """Return the PNG for a layer, rendering and caching it on first request.

        Parameters
        ----------
        job_dir : Path
            Directory in which rendered previews are cached.
        layer_name : str
            Identifier accepted by :func:`parse_layer_name`.
        lr_path, sr_path : Path
            Source rasters for the observed 10 m and super-resolved 2.5 m products.
        confidence_path : Path, optional
            Single-band confidence raster, required only for analysis layers.

        Returns
        -------
        Path
            Location of the cached PNG.

        Raises
        ------
        ValueError
            If the layer name is not servable.
        FileNotFoundError
            If the source raster needed to build the layer is missing.
        """
        job_dir = Path(job_dir)
        cached = job_dir / f"{layer_name}.png"
        if cached.is_file():
            return cached

        source, product, kind = parse_layer_name(layer_name)

        if kind == "analysis":
            return self._render_analysis_layer(job_dir, product, confidence_path, cached)

        stack, reference = self._load_stack(source, lr_path, sr_path)

        if kind == "index":
            arr = compute_index(stack, product, band_names=list(S2_10BAND_NAMES))
            return render_index_png(arr, product, cached)

        # Composite: reuse the calibrated renderers already used for RGB/CIR previews.
        from ntro_srm.web.services.sr_service import render_composite

        idx = COMPOSITES[product]["bands"]
        image = render_composite(stack, reference, idx)
        from PIL import Image

        Image.fromarray(image).save(cached, format="PNG", optimize=True)
        return cached

    def _render_analysis_layer(
        self,
        job_dir: Path,
        product: str,
        confidence_path: Optional[Path],
        cached: Path,
    ) -> Path:
        """Render a confidence / novelty / spread layer from the stored confidence raster."""
        if confidence_path is None or not Path(confidence_path).is_file():
            raise FileNotFoundError(
                f"Analysis layer '{product}' is unavailable: no confidence raster was produced "
                f"for this job."
            )
        with rasterio.open(confidence_path) as src:
            arr = src.read(1).astype(np.float32)
        cmap = ANALYSIS_LAYERS[product]["cmap"]
        vmin, vmax = (0.0, 1.0) if product == "confidence" else (None, None)
        rgba = apply_colormap_rgba(arr, cmap, vmin=vmin, vmax=vmax)
        from PIL import Image

        Image.fromarray(rgba, mode="RGBA").save(cached, format="PNG", optimize=True)
        return cached

    @staticmethod
    def _load_stack(source: str, lr_path: Path, sr_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load the 10-band stack for a layer source plus the LR radiometric reference."""
        lr = _read_reflectance(lr_path)
        if source == "lr":
            return lr, lr
        sr = _read_reflectance(sr_path)
        if source == "sr":
            return sr, lr
        # Bicubic baseline at the super-resolved grid.
        t = torch.from_numpy(lr).unsqueeze(0)
        up = F.interpolate(
            t, size=(sr.shape[1], sr.shape[2]), mode="bicubic", align_corners=False
        )
        return torch.clamp(up.squeeze(0), 0.0, 1.0).numpy(), lr

    # ------------------------------------------------------------------
    # Pixel probe
    # ------------------------------------------------------------------
    def probe_pixel(
        self,
        lat: float,
        lon: float,
        lr_path: Path,
        sr_path: Path,
        confidence_path: Optional[Path] = None,
        uncertainty_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Sample the observed and super-resolved spectra at a geographic coordinate.

        Parameters
        ----------
        lat, lon : float
            WGS84 coordinate of the point of interest.
        lr_path, sr_path : Path
            Observed 10 m and super-resolved 2.5 m rasters.
        confidence_path : Path, optional
            Confidence raster used to report per-pixel reliability.
        uncertainty_meta : dict, optional
            Scalar uncertainty summary used to fill in the risk band.

        Returns
        -------
        dict
            Payload consumed by the web UI's spectral inspector.

        Raises
        ------
        ValueError
            If the coordinate falls outside the super-resolved patch.
        """
        with rasterio.open(sr_path) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = float(xs[0]), float(ys[0])
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError("The selected point lies outside the super-resolved patch.")
            sr_values = _sample_window(src, row, col)
            crs_str = str(src.crs)

        with rasterio.open(lr_path) as src:
            lr_row, lr_col = src.index(x, y)
            lr_row = int(np.clip(lr_row, 0, src.height - 1))
            lr_col = int(np.clip(lr_col, 0, src.width - 1))
            lr_values = _sample_window(src, lr_row, lr_col)

        confidence_block: Optional[dict[str, Any]] = None
        if confidence_path is not None and Path(confidence_path).is_file():
            try:
                with rasterio.open(confidence_path) as src:
                    c_row, c_col = src.index(x, y)
                    if 0 <= c_row < src.height and 0 <= c_col < src.width:
                        value = float(src.read(1, window=((c_row, c_row + 1), (c_col, c_col + 1)))[0, 0])
                        confidence_block = {
                            "confidence": value,
                            "std": float((uncertainty_meta or {}).get("mean_std", float("nan"))),
                            "novelty": float((uncertainty_meta or {}).get("mean_novelty", float("nan"))),
                            "risk": _risk_from_confidence(value),
                        }
            except Exception:
                confidence_block = None

        indices: list[dict[str, Any]] = []
        for key, spec in INDEX_REGISTRY.items():
            sr_value = _index_from_spectrum(sr_values, key)
            lr_value = _index_from_spectrum(lr_values, key)
            label, color = _classify_scalar(sr_value, spec)
            indices.append(
                {
                    "key": key,
                    "name": spec.name,
                    "lr": lr_value,
                    "sr": sr_value,
                    "unit": "",
                    "class_label": label,
                    "class_color": color,
                }
            )

        return {
            "lat": float(lat),
            "lon": float(lon),
            "easting": x,
            "northing": y,
            "crs": crs_str,
            "row": int(row),
            "col": int(col),
            "band_names": list(S2_10BAND_NAMES),
            "wavelengths_nm": list(S2_WAVELENGTHS_NM),
            "lr": {"reflectance": [_clean(v) for v in lr_values], "row": int(lr_row), "col": int(lr_col)},
            "sr": {"reflectance": [_clean(v) for v in sr_values], "row": int(row), "col": int(col)},
            "bicubic": None,
            "indices": indices,
            "uncertainty": confidence_block,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a ``(C, H, W)`` float32 numpy view of a tensor or array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_dict(obj: Any) -> Optional[dict[str, Any]]:
    """Normalise a dataclass result, a dict, or ``None`` to a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _read_reflectance(path: Path) -> np.ndarray:
    """Read a 10-band raster as float reflectance in ``[0, 1]``."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] < 10:
        raise ValueError(f"Expected a 10-band raster, found {arr.shape[0]} bands in {path.name}.")
    arr = arr[:10]
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanmax(finite)) > 2.0:
        arr = arr / 10000.0
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_window(src: rasterio.DatasetReader, row: int, col: int) -> list[float]:
    """Read the full band stack at a single pixel."""
    window = ((row, row + 1), (col, col + 1))
    values = src.read(window=window).astype(np.float64)[:, 0, 0]
    if values.size and float(np.nanmax(values)) > 2.0:
        values = values / 10000.0
    return [float(v) for v in values[:10]]


def _clean(value: float) -> Optional[float]:
    """Convert a non-finite float to ``None`` so it survives JSON encoding."""
    return float(value) if np.isfinite(value) else None


def _index_from_spectrum(values: list[float], key: str) -> Optional[float]:
    """Evaluate a registered spectral index for a single 10-band spectrum."""
    if len(values) < 10:
        return None
    stack = np.asarray(values, dtype=np.float32).reshape(10, 1, 1)
    try:
        result = float(compute_index(stack, key, band_names=list(S2_10BAND_NAMES))[0, 0])
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _classify_scalar(value: Optional[float], spec: Any) -> tuple[Optional[str], Optional[str]]:
    """Resolve an index value to its interpretation class label and colour."""
    if value is None or not getattr(spec, "classes", ()):
        return (None, None)
    for lo, hi, label, color in spec.classes:
        if lo <= value <= hi:
            return (label, color)
    return (None, None)


def _risk_from_confidence(value: float) -> str:
    """Map a confidence value in ``[0, 1]`` onto the four reported risk bands."""
    score = value * 100.0
    if score >= 85.0:
        return "low"
    if score >= 70.0:
        return "moderate"
    if score >= 55.0:
        return "elevated"
    return "high"


def _write_single_band_geotiff(
    path: Path,
    array: np.ndarray,
    transform: Any,
    crs: Any,
    description: str,
) -> Path:
    """Write a single-band float32 GeoTIFF sharing the super-resolved georeferencing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.float32)
    height, width = data.shape
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)
        dst.update_tags(PRODUCT="NTRO-SRM reconstruction confidence", RANGE="0-1")
    return path
