"""Hand-rolled colour lookup tables for NTRO-SRM raster products.

Matplotlib is deliberately not a dependency of this project, so the perceptual ramps used
for spectral-index and uncertainty overlays are rebuilt here by linear interpolation
between published anchor stops. Every table is a ``(256, 3)`` uint8 array indexed from the
low end of the data range to the high end.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

# Anchor stops: (position in [0, 1], (r, g, b)). Values follow the published control
# points of each ramp closely enough to be visually indistinguishable at 256 steps.
COLORMAP_STOPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "viridis": [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ],
    "inferno": [
        (0.00, (0, 0, 4)),
        (0.20, (66, 10, 104)),
        (0.40, (147, 38, 103)),
        (0.60, (221, 81, 58)),
        (0.80, (252, 165, 10)),
        (1.00, (252, 255, 164)),
    ],
    "magma": [
        (0.00, (0, 0, 4)),
        (0.20, (59, 15, 112)),
        (0.40, (140, 41, 129)),
        (0.60, (222, 73, 104)),
        (0.80, (254, 159, 109)),
        (1.00, (252, 253, 191)),
    ],
    "turbo": [
        (0.000, (48, 18, 59)),
        (0.077, (65, 69, 171)),
        (0.154, (70, 117, 237)),
        (0.231, (57, 162, 252)),
        (0.308, (27, 207, 212)),
        (0.385, (36, 236, 166)),
        (0.462, (97, 252, 108)),
        (0.538, (164, 252, 59)),
        (0.615, (209, 232, 52)),
        (0.692, (243, 198, 58)),
        (0.769, (254, 155, 45)),
        (0.846, (243, 99, 21)),
        (0.923, (203, 42, 4)),
        (1.000, (122, 4, 3)),
    ],
    # Diverging red -> yellow -> green. Used for confidence (green == trustworthy).
    "rdylgn": [
        (0.00, (165, 0, 38)),
        (0.20, (215, 48, 39)),
        (0.35, (244, 109, 67)),
        (0.50, (255, 255, 191)),
        (0.65, (166, 217, 106)),
        (0.80, (102, 189, 99)),
        (1.00, (26, 152, 80)),
    ],
    # Diverging brown -> white -> teal, for bare-soil style indices.
    "brbg": [
        (0.00, (84, 48, 5)),
        (0.25, (191, 129, 45)),
        (0.45, (223, 194, 125)),
        (0.50, (245, 245, 245)),
        (0.55, (128, 205, 193)),
        (0.75, (53, 151, 143)),
        (1.00, (0, 60, 48)),
    ],
    "blues": [
        (0.00, (247, 251, 255)),
        (0.25, (198, 219, 239)),
        (0.50, (107, 174, 214)),
        (0.75, (33, 113, 181)),
        (1.00, (8, 48, 107)),
    ],
    # Vegetation ramp: water/bare brown -> tan -> yellow -> green -> dark green.
    "ndvi": [
        (0.00, (120, 90, 60)),
        (0.30, (191, 158, 108)),
        (0.50, (222, 205, 140)),
        (0.62, (233, 227, 122)),
        (0.75, (150, 200, 90)),
        (0.88, (60, 158, 66)),
        (1.00, (14, 92, 38)),
    ],
    # Water ramp: dry tan -> pale cyan -> deep blue.
    "ndwi": [
        (0.00, (196, 170, 124)),
        (0.35, (226, 214, 186)),
        (0.50, (198, 232, 236)),
        (0.68, (108, 189, 219)),
        (0.85, (36, 118, 184)),
        (1.00, (8, 48, 107)),
    ],
    # Built-up ramp: vegetated dark green -> neutral grey -> magenta/red.
    "ndbi": [
        (0.00, (23, 92, 55)),
        (0.30, (120, 168, 120)),
        (0.50, (222, 222, 222)),
        (0.70, (214, 130, 158)),
        (0.85, (194, 62, 96)),
        (1.00, (128, 16, 48)),
    ],
    "gray": [
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ],
}

_DEFAULT_COLORMAP = "viridis"


def build_lut(
    stops: Sequence[tuple[float, tuple[int, int, int]]],
    n: int = 256,
) -> np.ndarray:
    """Interpolate anchor stops into a dense colour lookup table.

    Parameters
    ----------
    stops : sequence of (float, tuple[int, int, int])
        Anchor points as ``(position, (r, g, b))`` with positions in ``[0, 1]``. They are
        sorted internally, so the caller need not supply them in order.
    n : int, default=256
        Number of entries in the resulting table.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` uint8 lookup table.

    Raises
    ------
    ValueError
        If fewer than two stops are supplied or ``n`` is below two.
    """
    if len(stops) < 2:
        raise ValueError("A colormap needs at least two anchor stops.")
    if n < 2:
        raise ValueError("A colormap needs at least two entries.")

    ordered = sorted(stops, key=lambda s: s[0])
    positions = np.asarray([s[0] for s in ordered], dtype=np.float64)
    colours = np.asarray([s[1] for s in ordered], dtype=np.float64)

    x = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.float64)
    for channel in range(3):
        lut[:, channel] = np.interp(x, positions, colours[:, channel])
    return np.clip(np.rint(lut), 0, 255).astype(np.uint8)


@lru_cache(maxsize=32)
def get_colormap(name: str) -> np.ndarray:
    """Return the cached ``(256, 3)`` uint8 table for a named colormap.

    Parameters
    ----------
    name : str
        Colormap key. An unrecognised name falls back to ``"viridis"`` rather than raising,
        so a mis-registered index still renders something readable.

    Returns
    -------
    numpy.ndarray
        ``(256, 3)`` uint8 lookup table.
    """
    stops = COLORMAP_STOPS.get(str(name).lower(), COLORMAP_STOPS[_DEFAULT_COLORMAP])
    return build_lut(stops, 256)


def _resolve_range(
    arr: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    finite: np.ndarray,
) -> tuple[float, float]:
    """Choose display bounds, defaulting to the 2nd/98th percentile of finite values."""
    if vmin is None or vmax is None:
        if np.any(finite):
            values = arr[finite]
            auto_min = float(np.percentile(values, 2.0))
            auto_max = float(np.percentile(values, 98.0))
        else:
            auto_min, auto_max = 0.0, 1.0
        lo = auto_min if vmin is None else float(vmin)
        hi = auto_max if vmax is None else float(vmax)
    else:
        lo, hi = float(vmin), float(vmax)

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def apply_colormap(
    arr: np.ndarray,
    name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
    nodata_rgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Map a two-dimensional float array to an RGB image.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(H, W)`` array of values.
    name : str, default="viridis"
        Colormap key, resolved through :func:`get_colormap`.
    vmin, vmax : float, optional
        Display bounds. Each defaults independently to the 2nd or 98th percentile of the
        finite values.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` array; pixels where it is ``False`` are treated as no-data.
    nodata_rgb : tuple[int, int, int], default=(0, 0, 0)
        Colour written for non-finite or masked pixels.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 image.
    """
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 2:
        values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"apply_colormap expects a 2-D array, got shape {np.shape(arr)}.")

    finite = np.isfinite(values)
    if mask is not None:
        finite = finite & np.asarray(mask, dtype=bool)

    lo, hi = _resolve_range(values, vmin, vmax, finite)
    normalised = np.clip((np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo), 0.0, 1.0)

    lut = get_colormap(name)
    indices = np.clip(np.rint(normalised * 255.0), 0, 255).astype(np.int32)
    rgb = lut[indices]

    if not np.all(finite):
        rgb = rgb.copy()
        rgb[~finite] = np.asarray(nodata_rgb, dtype=np.uint8)
    return rgb


def apply_colormap_rgba(
    arr: np.ndarray,
    name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
    alpha: int = 255,
) -> np.ndarray:
    """Map a two-dimensional array to RGBA, making invalid pixels fully transparent.

    Transparent no-data is what map overlays need: a masked or non-finite pixel disappears
    rather than painting a black hole over the basemap.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(H, W)`` array of values.
    name : str, default="viridis"
        Colormap key.
    vmin, vmax : float, optional
        Display bounds; see :func:`apply_colormap`.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` validity mask.
    alpha : int, default=255
        Opacity applied to valid pixels, clamped to ``[0, 255]``.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 4)`` uint8 image with ``alpha = 0`` on invalid pixels.
    """
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 2:
        values = np.squeeze(values)
    finite = np.isfinite(values)
    if mask is not None:
        finite = finite & np.asarray(mask, dtype=bool)

    rgb = apply_colormap(arr, name=name, vmin=vmin, vmax=vmax, mask=mask, nodata_rgb=(0, 0, 0))
    rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(finite, int(np.clip(alpha, 0, 255)), 0).astype(np.uint8)
    return rgba


def colormap_stops_hex(name: str, n: int = 9) -> list[str]:
    """Sample a colormap into evenly spaced hex strings for a web legend gradient.

    Parameters
    ----------
    name : str
        Colormap key.
    n : int, default=9
        Number of samples, clamped to at least two.

    Returns
    -------
    list[str]
        ``n`` colours as ``"#rrggbb"``, from the low end of the ramp to the high end.
    """
    count = max(2, int(n))
    lut = get_colormap(name)
    positions = np.linspace(0, 255, count).round().astype(int)
    return ["#{:02x}{:02x}{:02x}".format(*lut[p]) for p in positions]
