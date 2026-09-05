"""Spectral indices derived from super-resolved Sentinel-2 reflectance.

These are the application-layer products the problem statement calls for — crop
monitoring, urban mapping, water and burn assessment. Computed on the 2.5 m ten-band
stack they are genuinely new thematic products rather than a resampled version of the
10 m index, and :func:`index_delta_statistics` quantifies exactly how much extra
thematic edge detail the super-resolution actually delivered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ntro_srm.evaluation import colormaps as _cmaps
from ntro_srm.evaluation._common import safe_stats, to_chw_numpy
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


@dataclass(frozen=True)
class SpectralIndexSpec:
    """Definition and presentation metadata for one spectral index.

    Attributes
    ----------
    key : str
        Short lowercase identifier, e.g. ``"ndvi"``.
    name : str
        Human-readable name shown in the interface.
    formula : str
        The formula as an analyst would write it, using Sentinel-2 band names.
    bands : tuple[str, ...]
        Band names the index requires.
    cmap : str
        Colormap key from :mod:`ntro_srm.evaluation.colormaps`.
    vmin, vmax : float
        Fixed display bounds so the same index is comparable across scenes.
    description : str
        What the index measures.
    application : str
        The operational use this product serves.
    classes : tuple[tuple[float, float, str, str], ...]
        Interpretation classes as ``(low, high, label, "#rrggbb")``, contiguous across the
        full ``[vmin, vmax]`` domain.
    """

    key: str
    name: str
    formula: str
    bands: tuple[str, ...]
    cmap: str
    vmin: float
    vmax: float
    description: str
    application: str
    classes: tuple[tuple[float, float, str, str], ...] = ()


INDEX_REGISTRY: dict[str, SpectralIndexSpec] = {
    "ndvi": SpectralIndexSpec(
        key="ndvi",
        name="Normalised Difference Vegetation Index",
        formula="(B08 - B04) / (B08 + B04)",
        bands=("B08", "B04"),
        cmap="ndvi",
        vmin=-1.0,
        vmax=1.0,
        description="Canopy greenness and photosynthetic activity.",
        application="Crop vigour and within-field variability at 2.5 m.",
        classes=(
            (-1.0, 0.10, "Water / bare", "#8a6a3c"),
            (0.10, 0.30, "Sparse vegetation", "#bf9e6c"),
            (0.30, 0.50, "Moderate vegetation", "#96c85a"),
            (0.50, 1.0, "Dense vegetation", "#0e5c26"),
        ),
    ),
    "ndre": SpectralIndexSpec(
        key="ndre",
        name="Normalised Difference Red Edge",
        formula="(B08 - B05) / (B08 + B05)",
        bands=("B08", "B05"),
        cmap="ndvi",
        vmin=-1.0,
        vmax=1.0,
        description="Red-edge response, sensitive to chlorophyll content in dense canopies.",
        application="Crop nitrogen status and canopy stress detection.",
        classes=(
            (-1.0, 0.10, "Non-vegetated", "#8a6a3c"),
            (0.10, 0.25, "Low chlorophyll", "#d8c56b"),
            (0.25, 0.40, "Adequate", "#96c85a"),
            (0.40, 1.0, "High chlorophyll", "#0e5c26"),
        ),
    ),
    "savi": SpectralIndexSpec(
        key="savi",
        name="Soil-Adjusted Vegetation Index",
        formula="1.5 * (B08 - B04) / (B08 + B04 + 0.5)",
        bands=("B08", "B04"),
        cmap="ndvi",
        vmin=-1.0,
        vmax=1.0,
        description="Vegetation index with the soil-brightness term suppressed.",
        application="Crop monitoring over sparse canopies and exposed soil.",
        classes=(
            (-1.0, 0.10, "Bare soil", "#8a6a3c"),
            (0.10, 0.25, "Sparse cover", "#bf9e6c"),
            (0.25, 0.45, "Moderate cover", "#96c85a"),
            (0.45, 1.0, "Dense cover", "#0e5c26"),
        ),
    ),
    "evi": SpectralIndexSpec(
        key="evi",
        name="Enhanced Vegetation Index",
        formula="2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)",
        bands=("B08", "B04", "B02"),
        cmap="ndvi",
        vmin=-1.0,
        vmax=1.0,
        description="Vegetation index that resists saturation in dense canopy and corrects for aerosols.",
        application="Closed-canopy agriculture and forest condition.",
        classes=(
            (-1.0, 0.10, "Non-vegetated", "#8a6a3c"),
            (0.10, 0.30, "Sparse canopy", "#bf9e6c"),
            (0.30, 0.55, "Moderate canopy", "#96c85a"),
            (0.55, 1.0, "Dense canopy", "#0e5c26"),
        ),
    ),
    "ndwi": SpectralIndexSpec(
        key="ndwi",
        name="Normalised Difference Water Index",
        formula="(B03 - B08) / (B03 + B08)",
        bands=("B03", "B08"),
        cmap="ndwi",
        vmin=-1.0,
        vmax=1.0,
        description="Open-water response using the green and near-infrared bands.",
        application="Flood extent, shoreline delineation and disaster assessment.",
        classes=(
            (-1.0, -0.10, "Dry land", "#c4aa7c"),
            (-0.10, 0.10, "Moist soil / wetland", "#e2d6ba"),
            (0.10, 0.35, "Shallow water", "#6cbddb"),
            (0.35, 1.0, "Open water", "#08306b"),
        ),
    ),
    "mndwi": SpectralIndexSpec(
        key="mndwi",
        name="Modified Normalised Difference Water Index",
        formula="(B03 - B11) / (B03 + B11)",
        bands=("B03", "B11"),
        cmap="ndwi",
        vmin=-1.0,
        vmax=1.0,
        description="Water index using SWIR, which suppresses built-up false positives.",
        application="Water bodies in urban and mixed landscapes.",
        classes=(
            (-1.0, -0.10, "Dry land", "#c4aa7c"),
            (-0.10, 0.10, "Mixed / wet surface", "#e2d6ba"),
            (0.10, 0.40, "Shallow water", "#6cbddb"),
            (0.40, 1.0, "Open water", "#08306b"),
        ),
    ),
    "ndbi": SpectralIndexSpec(
        key="ndbi",
        name="Normalised Difference Built-up Index",
        formula="(B11 - B08) / (B11 + B08)",
        bands=("B11", "B08"),
        cmap="ndbi",
        vmin=-1.0,
        vmax=1.0,
        description="Contrast between SWIR and NIR that separates impervious surfaces from vegetation.",
        application="Urban mapping, settlement growth and infrastructure extent.",
        classes=(
            (-1.0, -0.15, "Vegetated", "#175c37"),
            (-0.15, 0.0, "Mixed / open", "#78a878"),
            (0.0, 0.20, "Built-up", "#d6829e"),
            (0.20, 1.0, "Dense built-up", "#801030"),
        ),
    ),
    "bsi": SpectralIndexSpec(
        key="bsi",
        name="Bare Soil Index",
        formula="((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))",
        bands=("B11", "B04", "B08", "B02"),
        cmap="brbg",
        vmin=-1.0,
        vmax=1.0,
        description="Separates exposed soil and rock from vegetated and water surfaces.",
        application="Land degradation, quarrying and post-harvest field state.",
        classes=(
            (-1.0, -0.20, "Water / dense vegetation", "#003c30"),
            (-0.20, 0.0, "Vegetated", "#80cdc1"),
            (0.0, 0.20, "Partially bare", "#dfc27d"),
            (0.20, 1.0, "Bare soil / rock", "#543005"),
        ),
    ),
    "nbr": SpectralIndexSpec(
        key="nbr",
        name="Normalised Burn Ratio",
        formula="(B08 - B12) / (B08 + B12)",
        bands=("B08", "B12"),
        cmap="rdylgn",
        vmin=-1.0,
        vmax=1.0,
        description="Contrast between NIR and long SWIR that responds strongly to combustion residue.",
        application="Wildfire burn severity and post-disaster damage mapping.",
        classes=(
            (-1.0, -0.10, "High severity", "#a50026"),
            (-0.10, 0.10, "Moderate severity", "#f46d43"),
            (0.10, 0.30, "Low severity", "#fee08b"),
            (0.30, 1.0, "Unburned / healthy", "#1a9850"),
        ),
    ),
    "ndmi": SpectralIndexSpec(
        key="ndmi",
        name="Normalised Difference Moisture Index",
        formula="(B08 - B11) / (B08 + B11)",
        bands=("B08", "B11"),
        cmap="blues",
        vmin=-1.0,
        vmax=1.0,
        description="Canopy water content from the NIR and short SWIR bands.",
        application="Drought stress, irrigation scheduling and fuel-moisture monitoring.",
        classes=(
            (-1.0, -0.20, "Very dry", "#f7fbff"),
            (-0.20, 0.0, "Dry", "#c6dbef"),
            (0.0, 0.25, "Moderate moisture", "#6baed6"),
            (0.25, 1.0, "High moisture", "#08306b"),
        ),
    ),
}


def _band_lookup(band_names: Sequence[str] | None) -> dict[str, int]:
    """Map band names to their position in the supplied stack ordering."""
    names = list(band_names) if band_names else list(S2_10BAND_NAMES)
    return {str(n).upper(): i for i, n in enumerate(names)}


def _require(stack: np.ndarray, lookup: dict[str, int], band: str, key: str) -> np.ndarray:
    """Fetch one band from the stack, raising a clean error when it is absent."""
    idx = lookup.get(band.upper())
    if idx is None or idx >= stack.shape[0]:
        raise ValueError(
            f"Spectral index '{key}' requires band {band}, which is not present in a "
            f"{stack.shape[0]}-band stack."
        )
    return stack[idx]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Normalised-difference division with divide-by-zero mapped to NaN rather than inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator == 0.0, np.nan, numerator / denominator)
    return np.asarray(out, dtype=np.float32)


def compute_index(
    stack: Any,
    key: str,
    band_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Evaluate one spectral index over a reflectance stack.

    Parameters
    ----------
    stack : numpy.ndarray or torch.Tensor
        Reflectance of shape ``(C, H, W)`` in ``[0, 1]``.
    key : str
        Registered index key, e.g. ``"ndvi"``.
    band_names : sequence of str, optional
        Band ordering of ``stack``. Defaults to the canonical 10-band Sentinel-2 order.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` float32 index. Pixels with a zero denominator are ``nan``, never ``inf``.

    Raises
    ------
    KeyError
        If ``key`` is not a registered index.
    ValueError
        If the stack is missing a band the index requires.
    """
    if key not in INDEX_REGISTRY:
        raise KeyError(
            f"Unknown spectral index '{key}'. Registered: {sorted(INDEX_REGISTRY)}."
        )
    spec = INDEX_REGISTRY[key]
    arr = to_chw_numpy(stack, dtype=np.float32)
    lookup = _band_lookup(band_names)

    def band(name: str) -> np.ndarray:
        return _require(arr, lookup, name, key)

    if key == "ndvi":
        nir, red = band("B08"), band("B04")
        return _ratio(nir - red, nir + red)
    if key == "ndre":
        nir, re1 = band("B08"), band("B05")
        return _ratio(nir - re1, nir + re1)
    if key == "savi":
        nir, red = band("B08"), band("B04")
        return _ratio(1.5 * (nir - red), nir + red + 0.5)
    if key == "evi":
        nir, red, blue = band("B08"), band("B04"), band("B02")
        return _ratio(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)
    if key == "ndwi":
        green, nir = band("B03"), band("B08")
        return _ratio(green - nir, green + nir)
    if key == "mndwi":
        green, swir1 = band("B03"), band("B11")
        return _ratio(green - swir1, green + swir1)
    if key == "ndbi":
        swir1, nir = band("B11"), band("B08")
        return _ratio(swir1 - nir, swir1 + nir)
    if key == "bsi":
        swir1, red, nir, blue = band("B11"), band("B04"), band("B08"), band("B02")
        return _ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    if key == "nbr":
        nir, swir2 = band("B08"), band("B12")
        return _ratio(nir - swir2, nir + swir2)
    if key == "ndmi":
        nir, swir1 = band("B08"), band("B11")
        return _ratio(nir - swir1, nir + swir1)

    raise KeyError(f"Spectral index '{key}' is registered but has no implementation.")


def compute_indices(
    stack: Any,
    keys: Sequence[str] | None = None,
    band_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate several indices over the same stack.

    Parameters
    ----------
    stack : numpy.ndarray or torch.Tensor
        Reflectance of shape ``(C, H, W)``.
    keys : sequence of str, optional
        Indices to compute. Defaults to every registered index.
    band_names : sequence of str, optional
        Band ordering of ``stack``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of index key to its ``(H, W)`` float32 raster.
    """
    selected = list(keys) if keys else list(INDEX_REGISTRY.keys())
    return {k: compute_index(stack, k, band_names=band_names) for k in selected}


def index_statistics(arr: np.ndarray, key: str) -> dict[str, Any]:
    """Summarise an index raster, including its interpretation-class breakdown.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(H, W)`` index raster.
    key : str
        Registered index key.

    Returns
    -------
    dict
        Distribution statistics plus a ``classes`` list of
        ``{"label", "color", "fraction", "pixels"}`` whose fractions sum to one over the
        valid pixels. All values are plain Python scalars, safe for JSON encoding.

    Raises
    ------
    KeyError
        If ``key`` is not a registered index.
    """
    if key not in INDEX_REGISTRY:
        raise KeyError(f"Unknown spectral index '{key}'.")
    spec = INDEX_REGISTRY[key]

    values = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(values)
    total = int(values.size)
    valid_count = int(finite.sum())
    stats = safe_stats(values)

    classes: list[dict[str, Any]] = []
    if valid_count:
        valid_values = values[finite]
        remaining = valid_count
        for position, (lo, hi, label, color) in enumerate(spec.classes):
            is_last = position == len(spec.classes) - 1
            # Half-open bins, with the final bin closed so the fractions sum to exactly one.
            if is_last:
                selected = (valid_values >= lo) & (valid_values <= hi)
            else:
                selected = (valid_values >= lo) & (valid_values < hi)
            count = int(selected.sum())
            remaining -= count
            classes.append(
                {
                    "label": label,
                    "color": color,
                    "fraction": count / valid_count,
                    "pixels": count,
                }
            )
        # Values below the first bin (only possible for out-of-domain input) are folded
        # into the lowest class so the reported fractions stay a true partition.
        if remaining > 0 and classes:
            classes[0]["pixels"] += remaining
            classes[0]["fraction"] = classes[0]["pixels"] / valid_count
    else:
        for lo, hi, label, color in spec.classes:
            classes.append({"label": label, "color": color, "fraction": 0.0, "pixels": 0})

    return {
        "key": spec.key,
        "name": spec.name,
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "p05": stats["p05"],
        "p50": stats["p50"],
        "p95": stats["p95"],
        "valid_fraction": (valid_count / total) if total else 0.0,
        "classes": classes,
    }


def render_index_png(arr: np.ndarray, key: str, path: Any) -> Path:
    """Write an index raster as a transparent-nodata RGBA PNG map overlay.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(H, W)`` index raster.
    key : str
        Registered index key, which supplies the colormap and fixed display bounds.
    path : str or Path
        Destination file. Parent directories are created if needed.

    Returns
    -------
    Path
        The written file path.

    Raises
    ------
    KeyError
        If ``key`` is not a registered index.
    """
    if key not in INDEX_REGISTRY:
        raise KeyError(f"Unknown spectral index '{key}'.")
    spec = INDEX_REGISTRY[key]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgba = _cmaps.apply_colormap_rgba(
        np.asarray(arr, dtype=np.float32),
        name=spec.cmap,
        vmin=spec.vmin,
        vmax=spec.vmax,
    )
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG", optimize=True)
    return out


def _laplacian_energy(arr: np.ndarray) -> float:
    """Mean absolute 3x3 Laplacian response, a scale-free measure of edge content."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    padded = np.pad(a, 1, mode="reflect")
    response = 8.0 * a
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            response = response - padded[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]]
    return float(np.abs(response).mean())


def index_delta_statistics(lr_arr: np.ndarray, sr_arr: np.ndarray, key: str) -> dict[str, float]:
    """Compare a native-resolution index against its super-resolved counterpart.

    The two rasters usually differ in size; the native index is bicubically resampled onto
    the super-resolved grid first so the comparison is like-for-like.

    Parameters
    ----------
    lr_arr : numpy.ndarray
        ``(h, w)`` index computed on the observed 10 m stack.
    sr_arr : numpy.ndarray
        ``(H, W)`` index computed on the 2.5 m stack.
    key : str
        Registered index key, retained for provenance in the returned payload.

    Returns
    -------
    dict[str, float]
        ``mean_lr``, ``mean_sr``, ``mean_abs_delta`` and ``edge_gain``. An ``edge_gain``
        above one means the super-resolved product carries more thematic edge detail than
        interpolation alone would produce; it is ``nan`` when the native index is flat.
    """
    import torch
    import torch.nn.functional as F

    sr = np.asarray(sr_arr, dtype=np.float32)
    lr = np.asarray(lr_arr, dtype=np.float32)

    if lr.shape != sr.shape:
        tensor = torch.from_numpy(np.nan_to_num(lr, nan=0.0)[None, None])
        resized = F.interpolate(
            tensor, size=(sr.shape[0], sr.shape[1]), mode="bicubic", align_corners=False
        )
        lr_on_grid = resized[0, 0].numpy()
    else:
        lr_on_grid = np.nan_to_num(lr, nan=0.0)

    finite = np.isfinite(sr)
    mean_sr = float(np.nanmean(sr)) if finite.any() else float("nan")
    mean_lr = float(np.nanmean(lr)) if np.isfinite(lr).any() else float("nan")
    delta = np.abs(np.nan_to_num(sr, nan=0.0) - lr_on_grid)
    mean_abs_delta = float(delta.mean()) if delta.size else float("nan")

    lr_energy = _laplacian_energy(lr_on_grid)
    sr_energy = _laplacian_energy(sr)
    edge_gain = float(sr_energy / lr_energy) if lr_energy > 1e-12 else float("nan")

    return {
        "key": key,
        "mean_lr": mean_lr,
        "mean_sr": mean_sr,
        "mean_abs_delta": mean_abs_delta,
        "edge_gain": edge_gain,
    }


def registry_as_json() -> list[dict[str, Any]]:
    """Serialise the index registry for the web interface.

    Returns
    -------
    list[dict]
        One entry per index carrying its definition, display bounds, interpretation
        classes and a nine-stop ``legend_hex`` gradient for the map legend.
    """
    entries: list[dict[str, Any]] = []
    for spec in INDEX_REGISTRY.values():
        entries.append(
            {
                "key": spec.key,
                "name": spec.name,
                "formula": spec.formula,
                "bands": list(spec.bands),
                "cmap": spec.cmap,
                "vmin": spec.vmin,
                "vmax": spec.vmax,
                "description": spec.description,
                "application": spec.application,
                "classes": [
                    {"lo": lo, "hi": hi, "label": label, "color": color}
                    for lo, hi, label, color in spec.classes
                ],
                "legend_hex": _cmaps.colormap_stops_hex(spec.cmap, 9),
            }
        )
    return entries
