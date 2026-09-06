"""Full-reference image quality metrics for Sentinel-2 super-resolution assessment.

Pure-numpy implementations of the standard remote-sensing accuracy measures used to
score a super-resolved product against a reference stack: PSNR, SSIM, RMSE, MAE, SAM,
ERGAS, UIQI/Q, SCC and the plain Pearson correlation coefficient.

No scipy and no skimage: the Gaussian window of SSIM and the Laplacian high-pass of
SCC are implemented as shift-and-accumulate convolutions over mirror-reflected edge
padding, vectorised across the band axis.

All public functions accept either ``np.ndarray`` or ``torch.Tensor`` in the canonical
(C, H, W) channel-first layout with reflectance in [0, 1].  Every statistic is taken
over the pixels that are finite in *both* inputs, so the metrics degrade gracefully on
masked / cloud-flagged scenes instead of collapsing to NaN.  ``float('nan')`` is
returned only where a value is genuinely undefined (e.g. no valid pixel at all, or a
Pearson correlation against a constant image).
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
import torch

from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES

# The shared converters live in ``_common`` (owned by a sibling module).  The import is
# tolerant so that this module stays importable and testable on its own; the local
# fallback below implements exactly the documented contract.
try:  # pragma: no cover - trivial import shim
    from ntro_srm.evaluation._common import to_chw_numpy as _shared_to_chw
except Exception:  # pragma: no cover - sibling module not available
    _shared_to_chw = None

ArrayLike = Union[np.ndarray, torch.Tensor]

# A perfect reconstruction has zero MSE and therefore infinite PSNR.  Infinity is not
# safely JSON round-trippable, so it is reported as this finite ceiling instead.
_PSNR_MAX_DB: float = 100.0

# Guard used for divisions by variances / means that are numerically zero.
_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Array plumbing
# ---------------------------------------------------------------------------


def _to_chw(x: ArrayLike, dtype: type = np.float64) -> np.ndarray:
    """Return `x` as a contiguous (C, H, W) float numpy array.

    Delegates to :func:`ntro_srm.evaluation._common.to_chw_numpy` when that module is
    available and falls back to an equivalent local conversion otherwise.
    """
    if _shared_to_chw is not None:
        arr = np.asarray(_shared_to_chw(x))
    elif isinstance(x, torch.Tensor):
        arr = x.detach().to("cpu").numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Batched input with batch size > 1 is not supported: {arr.shape}")
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected a 2-D, 3-D or (1, C, H, W) array, got shape {arr.shape}")

    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def _pair(pred: ArrayLike, ref: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert both inputs to (C, H, W) float64 and return them with the shared mask.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        ``(pred, ref, mask)`` where ``mask`` is the boolean (H, W) array of pixels that
        are finite in every band of both inputs.
    """
    p = _to_chw(pred)
    r = _to_chw(ref)
    if p.shape != r.shape:
        raise ValueError(f"pred and ref must have identical shapes, got {p.shape} vs {r.shape}")
    mask = np.isfinite(p).all(axis=0) & np.isfinite(r).all(axis=0)
    return p, r, mask


def _finite_pairs(pred: ArrayLike, ref: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pred, ref, mask)`` with every non-finite entry replaced by 0.0."""
    p, r, mask = _pair(pred, ref)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return p, r, mask


def _mean_or_nan(values: Sequence[float]) -> float:
    """Mean of the finite entries of `values`; NaN when none is finite."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _resolve_band_names(n_bands: int, band_names: Sequence[str] | None) -> list[str]:
    """Band labels for `n_bands` channels, defaulting to the Sentinel-2 10-band order."""
    if band_names is not None:
        names = [str(n) for n in band_names]
        if len(names) >= n_bands:
            return names[:n_bands]
        return names + [f"Band {i + 1}" for i in range(len(names), n_bands)]
    if n_bands == len(S2_10BAND_NAMES):
        return list(S2_10BAND_NAMES)
    return [f"Band {i + 1}" for i in range(n_bands)]


# ---------------------------------------------------------------------------
# Filtering primitives (no scipy)
# ---------------------------------------------------------------------------


def _reflect_indices(n: int, radius: int) -> np.ndarray:
    """Index vector of length ``n + 2 * radius`` implementing mirror ('reflect') padding.

    The edge sample is not duplicated, i.e. ``... c b | a b c d | c b ...``, matching
    ``np.pad(mode='reflect')`` while additionally supporting a radius larger than `n`.
    """
    if n <= 1:
        return np.zeros(n + 2 * radius, dtype=np.intp)
    period = 2 * n - 2
    idx = np.mod(np.arange(-radius, n + radius), period)
    return np.where(idx >= n, period - idx, idx).astype(np.intp)


def _convolve1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve `arr` with a 1-D `kernel` along `axis` using reflected padding.

    The kernel taps are accumulated in place into a single preallocated buffer, which
    keeps the memory traffic of the SSIM window down on full-size super-resolved scenes.
    """
    radius = (kernel.size - 1) // 2
    n = arr.shape[axis]
    padded = np.take(arr, _reflect_indices(n, radius), axis=axis)
    out = np.zeros(arr.shape, dtype=np.float64)
    tmp = np.empty(arr.shape, dtype=np.float64)
    slicer: list[slice] = [slice(None)] * arr.ndim
    for tap in range(kernel.size):
        slicer[axis] = slice(tap, tap + n)
        np.multiply(padded[tuple(slicer)], kernel[tap], out=tmp)
        out += tmp
    return out


def _gaussian_kernel1d(sigma: float, truncate: float) -> np.ndarray:
    """Normalised 1-D Gaussian of radius ``int(truncate * sigma + 0.5)``."""
    sigma = max(float(sigma), _EPS)
    radius = max(int(truncate * sigma + 0.5), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    total = float(kernel.sum())
    return kernel / total if total > 0.0 else np.full(kernel.shape, 1.0 / kernel.size)


def _gaussian_blur(arr: np.ndarray, sigma: float = 1.5, truncate: float = 3.5) -> np.ndarray:
    """Separable Gaussian blur of a (C, H, W) array along H then W ('reflect' edges)."""
    kernel = _gaussian_kernel1d(sigma, truncate)
    blurred = _convolve1d(arr, kernel, axis=-2)
    return _convolve1d(blurred, kernel, axis=-1)


def _box3_sum(arr: np.ndarray) -> np.ndarray:
    """Sum of each 3x3 neighbourhood of a (C, H, W) array, with reflected edges."""
    out = arr
    for axis in (-2, -1):
        n = out.shape[axis]
        padded = np.take(out, _reflect_indices(n, 1), axis=axis)
        slicer: list[slice] = [slice(None)] * out.ndim
        slicer[axis] = slice(0, n)
        acc = np.array(padded[tuple(slicer)], dtype=np.float64, copy=True)
        for tap in (1, 2):
            slicer[axis] = slice(tap, tap + n)
            acc += padded[tuple(slicer)]
        out = acc
    return out


def _laplacian_highpass(arr: np.ndarray) -> np.ndarray:
    """3x3 Laplacian high-pass ``[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]`` of a (C, H, W) array.

    The kernel equals ``9 * delta - ones((3, 3))``, so the response is evaluated as
    ``9 * arr - box3(arr)`` with a separable (and therefore cheaper) box sum, rather
    than as nine weighted shifts.
    """
    return 9.0 * arr - _box3_sum(arr)


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrink a boolean (H, W) mask by `radius` pixels (separable minimum filter)."""
    if radius <= 0 or mask.all():
        return mask
    out = mask
    for axis in (0, 1):
        n = out.shape[axis]
        padded = np.take(out, _reflect_indices(n, radius), axis=axis)
        acc = np.ones(out.shape, dtype=bool)
        slicer: list[slice] = [slice(None)] * 2
        for tap in range(2 * radius + 1):
            slicer[axis] = slice(tap, tap + n)
            acc &= padded[tuple(slicer)]
        out = acc
    return out


def _fill_invalid(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the pixels outside `mask` (per band) with that band's valid mean.

    Filling with a constant keeps the separable convolutions finite; the contaminated
    pixels are excluded again afterwards via an eroded mask.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    if mask.all():
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    invalid = ~mask
    for band in range(out.shape[0]):
        valid_values = out[band][mask]
        valid_values = valid_values[np.isfinite(valid_values)]
        out[band][invalid] = float(valid_values.mean()) if valid_values.size else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two flat finite vectors; NaN when either is constant."""
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    if denom <= _EPS:
        return float("nan")
    return float(float(a @ b) / denom)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def psnr(pred: ArrayLike, ref: ArrayLike, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio over all bands, in decibels.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    data_range : float, default=1.0
        Dynamic range of the data (1.0 for [0, 1] reflectance).

    Returns
    -------
    float
        ``10 * log10(data_range ** 2 / MSE)`` over the pixels valid in both inputs.
        A perfect match is reported as 100.0 dB rather than infinity, so the value stays
        JSON-serialisable; NaN when no valid pixel exists.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return float("nan")
    diff = (p - r)[:, mask]
    mse = float(np.mean(diff * diff))
    if mse <= _EPS * _EPS:
        return _PSNR_MAX_DB
    return float(min(10.0 * math.log10((float(data_range) ** 2) / mse), _PSNR_MAX_DB))


def band_psnr(pred: ArrayLike, ref: ArrayLike, data_range: float = 1.0) -> list[float]:
    """Per-band PSNR in decibels.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    data_range : float, default=1.0
        Dynamic range of the data.

    Returns
    -------
    list of float
        One PSNR value per band, in band order.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return [float("nan")] * p.shape[0]
    out: list[float] = []
    for band in range(p.shape[0]):
        diff = (p[band] - r[band])[mask]
        mse = float(np.mean(diff * diff))
        if mse <= _EPS * _EPS:
            out.append(_PSNR_MAX_DB)
        else:
            out.append(float(min(10.0 * math.log10((float(data_range) ** 2) / mse), _PSNR_MAX_DB)))
    return out


def rmse(pred: ArrayLike, ref: ArrayLike) -> float:
    """Root-mean-square error over all bands, in reflectance units.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.

    Returns
    -------
    float
        RMSE over the pixels valid in both inputs; NaN when none is valid.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return float("nan")
    diff = (p - r)[:, mask]
    return float(math.sqrt(float(np.mean(diff * diff))))


def band_rmse(pred: ArrayLike, ref: ArrayLike) -> list[float]:
    """Per-band root-mean-square error.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.

    Returns
    -------
    list of float
        One RMSE value per band, in band order.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return [float("nan")] * p.shape[0]
    diff = (p - r)[:, mask]
    return [float(math.sqrt(float(np.mean(row * row)))) for row in diff]


def mae(pred: ArrayLike, ref: ArrayLike) -> float:
    """Mean absolute error over all bands, in reflectance units.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.

    Returns
    -------
    float
        MAE over the pixels valid in both inputs; NaN when none is valid.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((p - r)[:, mask])))


# ---------------------------------------------------------------------------
# Structural similarity
# ---------------------------------------------------------------------------


def _ssim_maps(
    pred: ArrayLike,
    ref: ArrayLike,
    data_range: float,
    sigma: float,
    truncate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-band SSIM map (C, H, W) and the (H, W) mask to average it over."""
    p, r, mask = _pair(pred, ref)
    if not mask.any():
        return np.full(p.shape, np.nan), np.zeros(mask.shape, dtype=bool)

    x = _fill_invalid(p, mask)
    y = _fill_invalid(r, mask)

    radius = (_gaussian_kernel1d(sigma, truncate).size - 1) // 2

    mu_x = _gaussian_blur(x, sigma, truncate)
    mu_y = _gaussian_blur(y, sigma, truncate)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = _gaussian_blur(x * x, sigma, truncate) - mu_xx
    sigma_yy = _gaussian_blur(y * y, sigma, truncate) - mu_yy
    sigma_xy = _gaussian_blur(x * y, sigma, truncate) - mu_xy

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    ssim_map = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=np.abs(denominator) > _EPS,
    )

    # Pixels whose Gaussian window overlapped a filled (invalid) pixel carry fabricated
    # local statistics, so the mask is eroded by the window radius before averaging.
    eroded = _erode_mask(mask, radius)
    if not eroded.any():
        eroded = mask
    return ssim_map, eroded


def ssim(
    pred: ArrayLike,
    ref: ArrayLike,
    data_range: float = 1.0,
    sigma: float = 1.5,
    truncate: float = 3.5,
) -> float:
    """Mean Gaussian-windowed structural similarity (Wang et al. 2004) over all bands.

    The local statistics are weighted by an isotropic Gaussian of standard deviation
    `sigma`, truncated at a radius of ``int(truncate * sigma + 0.5)`` pixels and applied
    as two separable 1-D convolutions over mirror-reflected edges.  The stabilisers are
    ``C1 = (0.01 * data_range) ** 2`` and ``C2 = (0.03 * data_range) ** 2`` (K1 = 0.01,
    K2 = 0.03).

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    data_range : float, default=1.0
        Dynamic range of the data (1.0 for [0, 1] reflectance).
    sigma : float, default=1.5
        Standard deviation of the Gaussian window, in pixels.
    truncate : float, default=3.5
        Window radius in units of `sigma`.

    Returns
    -------
    float
        Mean of the SSIM map over every band and every valid pixel, in [-1, 1] and
        exactly 1.0 for identical inputs; NaN when no valid pixel exists.
    """
    ssim_map, mask = _ssim_maps(pred, ref, data_range, sigma, truncate)
    if not mask.any():
        return float("nan")
    values = ssim_map[:, mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def band_ssim(
    pred: ArrayLike,
    ref: ArrayLike,
    data_range: float = 1.0,
    sigma: float = 1.5,
    truncate: float = 3.5,
) -> list[float]:
    """Per-band Gaussian-windowed SSIM.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    data_range : float, default=1.0
        Dynamic range of the data.
    sigma : float, default=1.5
        Standard deviation of the Gaussian window, in pixels.
    truncate : float, default=3.5
        Window radius in units of `sigma`.

    Returns
    -------
    list of float
        One SSIM value per band, in band order.
    """
    ssim_map, mask = _ssim_maps(pred, ref, data_range, sigma, truncate)
    if not mask.any():
        return [float("nan")] * ssim_map.shape[0]
    out: list[float] = []
    for band in range(ssim_map.shape[0]):
        values = ssim_map[band][mask]
        values = values[np.isfinite(values)]
        out.append(float(values.mean()) if values.size else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------


def _sam_angles(pred: ArrayLike, ref: ArrayLike, degrees: bool) -> np.ndarray:
    """Float64 (H, W) spectral-angle map; NaN where the angle is undefined."""
    p, r, mask = _finite_pairs(pred, ref)
    dot = np.sum(p * r, axis=0)
    norm_p = np.sqrt(np.sum(p * p, axis=0))
    norm_r = np.sqrt(np.sum(r * r, axis=0))
    usable = mask & (norm_p > _EPS) & (norm_r > _EPS)

    out = np.full(dot.shape, np.nan, dtype=np.float64)
    if usable.any():
        cosine = dot[usable] / (norm_p[usable] * norm_r[usable])
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        out[usable] = np.degrees(angle) if degrees else angle
    return out


def sam_map(pred: ArrayLike, ref: ArrayLike, degrees: bool = True) -> np.ndarray:
    """Per-pixel Spectral Angle Mapper between the predicted and reference spectra.

    The angle between the two C-dimensional reflectance vectors of a pixel is
    ``arccos(<p, r> / (||p|| * ||r||))``, with the cosine clipped into [-1, 1] before the
    arc-cosine to absorb floating-point overshoot.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    degrees : bool, default=True
        Return degrees instead of radians.

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W), sized for display and overlay export.  Pixels
        that are invalid, or whose spectrum has a numerically zero norm in either input
        (the angle is then undefined), are NaN.
    """
    return _sam_angles(pred, ref, degrees).astype(np.float32)


def sam(pred: ArrayLike, ref: ArrayLike, degrees: bool = True) -> float:
    """Mean Spectral Angle Mapper over the pixels with a non-zero spectrum.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    degrees : bool, default=True
        Return degrees instead of radians.

    Returns
    -------
    float
        Mean spectral angle (lower is better, 0 for identical spectra); NaN when no pixel
        has a usable spectrum in both inputs.  Accumulated in float64, so it is not
        limited by the float32 precision of :func:`sam_map`.
    """
    angles = _sam_angles(pred, ref, degrees)
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def ergas(pred: ArrayLike, ref: ArrayLike, ratio: float = 4.0) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese (Wald 2002).

    Convention used here::

        ERGAS = (100 / ratio) * sqrt( mean_b( (RMSE_b / mean_ref_b) ** 2 ) )

    `ratio` is the **upscaling factor**, i.e. the ratio of the coarse pixel size to the
    fine pixel size, so a 4x super-resolution (10 m -> 2.5 m) is scored with
    ``ratio=4.0`` and the leading constant is the standard Wald term
    ``100 * h / l = 100 / 4 = 25``.  ``mean_ref_b`` is the mean of the *reference* band
    over the valid pixels; a band whose reference mean is numerically zero is excluded,
    because its relative error is undefined.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    ratio : float, default=4.0
        Spatial upscaling factor between the coarse and the fine grid.

    Returns
    -------
    float
        ERGAS in percent (lower is better, 0 for a perfect reconstruction); NaN when no
        band has a usable reference mean.
    """
    p, r, mask = _finite_pairs(pred, ref)
    ratio = float(ratio)
    if not mask.any() or abs(ratio) <= _EPS:
        return float("nan")

    terms: list[float] = []
    for band in range(p.shape[0]):
        ref_band = r[band][mask]
        mu = float(ref_band.mean())
        if abs(mu) <= _EPS:
            continue
        diff = p[band][mask] - ref_band
        terms.append((math.sqrt(float(np.mean(diff * diff))) / mu) ** 2)

    if not terms:
        return float("nan")
    return float((100.0 / ratio) * math.sqrt(float(np.mean(terms))))


# ---------------------------------------------------------------------------
# Universal Image Quality Index
# ---------------------------------------------------------------------------


def _block_sums(arr: np.ndarray, block: int) -> np.ndarray:
    """Sum an (H, W) array over non-overlapping ``block x block`` tiles."""
    height, width = arr.shape
    n_rows, n_cols = height // block, width // block
    trimmed = arr[: n_rows * block, : n_cols * block]
    return trimmed.reshape(n_rows, block, n_cols, block).sum(axis=(1, 3))


def q_index_per_band(pred: ArrayLike, ref: ArrayLike, block_size: int = 8) -> list[float]:
    """Per-band Universal Image Quality Index (Wang & Bovik 2002).

    The index is evaluated on non-overlapping ``block_size x block_size`` tiles and
    averaged over those holding at least four valid pixels::

        Q = (4 * sigma_xy * mu_x * mu_y) / ((sigma_x^2 + sigma_y^2) * (mu_x^2 + mu_y^2))

    The second-order moments appear once in the numerator and once in the denominator, so
    the result does not depend on whether the biased or the unbiased variance estimator is
    used.  Partial tiles at the right/bottom edge are dropped; an image smaller than
    `block_size` is treated as a single tile.  A tile whose denominator vanishes (both
    patches flat) scores 1.0 when the two patches are identical and 0.0 otherwise.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    block_size : int, default=8
        Side length of the square analysis tile, in pixels.

    Returns
    -------
    list of float
        One Q value per band in [-1, 1] (1.0 for identical inputs); NaN for a band with
        no usable tile.
    """
    p, r, mask = _finite_pairs(pred, ref)
    n_bands, height, width = p.shape
    block = int(block_size) if block_size and int(block_size) > 0 else max(height, width)
    block = max(1, min(block, height, width))

    weight = mask.astype(np.float64)
    counts = _block_sums(weight, block)
    usable = counts >= 4.0
    if not usable.any():
        return [float("nan")] * n_bands

    n_valid = counts[usable]
    out: list[float] = []
    for band in range(n_bands):
        x = p[band] * weight
        y = r[band] * weight
        delta = p[band] - r[band]

        sum_x = _block_sums(x, block)[usable]
        sum_y = _block_sums(y, block)[usable]
        sum_xx = _block_sums(x * p[band], block)[usable]
        sum_yy = _block_sums(y * r[band], block)[usable]
        sum_xy = _block_sums(x * r[band], block)[usable]
        sum_dd = _block_sums(delta * delta * weight, block)[usable]

        mu_x = sum_x / n_valid
        mu_y = sum_y / n_valid
        var_x = sum_xx / n_valid - mu_x * mu_x
        var_y = sum_yy / n_valid - mu_y * mu_y
        cov_xy = sum_xy / n_valid - mu_x * mu_y

        numerator = 4.0 * cov_xy * mu_x * mu_y
        denominator = (var_x + var_y) * (mu_x * mu_x + mu_y * mu_y)
        degenerate = np.abs(denominator) <= _EPS
        q = np.divide(
            numerator,
            denominator,
            out=np.zeros(numerator.shape, dtype=np.float64),
            where=~degenerate,
        )
        # A flat, exactly matching tile is a perfect reconstruction; a flat mismatching
        # tile carries no structure to correlate and scores zero.
        q = np.where(degenerate, np.where(sum_dd / n_valid <= 1e-20, 1.0, 0.0), q)
        out.append(float(q.mean()) if q.size else float("nan"))
    return out


def uiqi(pred: ArrayLike, ref: ArrayLike, block_size: int = 8) -> float:
    """Universal Image Quality Index averaged over blocks and bands.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    block_size : int, default=8
        Side length of the square analysis tile, in pixels.

    Returns
    -------
    float
        Mean Q over all bands in [-1, 1] (1.0 for identical inputs); NaN when no band
        yields a usable tile.
    """
    return _mean_or_nan(q_index_per_band(pred, ref, block_size=block_size))


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------


def scc(pred: ArrayLike, ref: ArrayLike) -> float:
    """Spatial Correlation Coefficient of the 3x3 Laplacian high-pass responses.

    Both stacks are filtered with ``[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]`` over mirror-
    reflected edges, and the two responses are Pearson-correlated per band and averaged.
    It measures how faithfully the fine spatial structure (edges, texture) is reproduced,
    independently of any radiometric offset or gain.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.

    Returns
    -------
    float
        Mean correlation in [-1, 1] (1.0 for identical inputs); NaN when no band has a
        non-constant high-pass response.
    """
    p, r, mask = _pair(pred, ref)
    if not mask.any():
        return float("nan")

    hp_pred = _laplacian_highpass(_fill_invalid(p, mask))
    hp_ref = _laplacian_highpass(_fill_invalid(r, mask))

    # Drop the one-pixel rim around invalid data, whose response used filled values.
    inner = _erode_mask(mask, 1)
    if not inner.any():
        inner = mask

    per_band = [_pearson(hp_pred[band][inner], hp_ref[band][inner]) for band in range(p.shape[0])]
    return _mean_or_nan(per_band)


def correlation(pred: ArrayLike, ref: ArrayLike) -> float:
    """Pearson correlation coefficient pooled over every band and valid pixel.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.

    Returns
    -------
    float
        Correlation in [-1, 1] (1.0 for identical inputs); NaN when either input is
        constant over the valid pixels, where the coefficient is undefined.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return float("nan")
    return _pearson(p[:, mask].ravel(), r[:, mask].ravel())


def _band_correlation(pred: ArrayLike, ref: ArrayLike) -> list[float]:
    """Per-band Pearson correlation coefficient.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.

    Returns
    -------
    list of float
        One correlation value per band, in band order.
    """
    p, r, mask = _finite_pairs(pred, ref)
    if not mask.any():
        return [float("nan")] * p.shape[0]
    return [_pearson(p[band][mask], r[band][mask]) for band in range(p.shape[0])]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_all(
    pred: ArrayLike,
    ref: ArrayLike,
    data_range: float = 1.0,
    ratio: float = 4.0,
    band_names: Sequence[str] | None = None,
) -> dict:
    """Compute the full accuracy-assessment suite in a single call.

    Parameters
    ----------
    pred : np.ndarray or torch.Tensor
        Predicted stack of shape (C, H, W).
    ref : np.ndarray or torch.Tensor
        Reference stack of the same shape.
    data_range : float, default=1.0
        Dynamic range used by PSNR and SSIM.
    ratio : float, default=4.0
        Spatial upscaling factor used by ERGAS.
    band_names : sequence of str, optional
        Band labels; defaults to :data:`S2_10BAND_NAMES` for a 10-band stack and to
        ``["Band 1", ...]`` otherwise.

    Returns
    -------
    dict
        JSON-serialisable dictionary with exactly the keys ``psnr_db``, ``ssim``,
        ``rmse``, ``mae``, ``sam_deg``, ``ergas``, ``uiqi``, ``scc``, ``cc``,
        ``n_bands``, ``shape`` (``[C, H, W]``) and ``per_band`` (one
        ``{"band", "psnr_db", "ssim", "rmse", "cc"}`` entry per band).
    """
    p = _to_chw(pred)
    r = _to_chw(ref)
    if p.shape != r.shape:
        raise ValueError(f"pred and ref must have identical shapes, got {p.shape} vs {r.shape}")

    n_bands, height, width = p.shape
    names = _resolve_band_names(n_bands, band_names)

    psnr_bands = band_psnr(p, r, data_range=data_range)
    ssim_bands = band_ssim(p, r, data_range=data_range)
    rmse_bands = band_rmse(p, r)
    cc_bands = _band_correlation(p, r)

    per_band = [
        {
            "band": names[i],
            "psnr_db": float(psnr_bands[i]),
            "ssim": float(ssim_bands[i]),
            "rmse": float(rmse_bands[i]),
            "cc": float(cc_bands[i]),
        }
        for i in range(n_bands)
    ]

    return {
        "psnr_db": float(psnr(p, r, data_range=data_range)),
        "ssim": float(_mean_or_nan(ssim_bands)),
        "rmse": float(rmse(p, r)),
        "mae": float(mae(p, r)),
        "sam_deg": float(sam(p, r, degrees=True)),
        "ergas": float(ergas(p, r, ratio=ratio)),
        "uiqi": float(uiqi(p, r)),
        "scc": float(scc(p, r)),
        "cc": float(correlation(p, r)),
        "n_bands": int(n_bands),
        "shape": [int(n_bands), int(height), int(width)],
        "per_band": per_band,
    }


# Presentation metadata consumed by the web UI.  ``good`` / ``excellent`` are the
# thresholds used to colour a metric chip; for 'lower is better' metrics ``good`` is
# numerically greater than ``excellent``.  The values are calibrated for 4x
# multispectral Sentinel-2 super-resolution scored under Wald's protocol on [0, 1]
# surface reflectance.
METRIC_META: dict[str, dict] = {
    "psnr_db": {
        "label": "PSNR",
        "unit": "dB",
        "better": "higher",
        "good": 28.0,
        "excellent": 34.0,
        "description": (
            "Peak signal-to-noise ratio between the reconstruction and the reference. "
            "A global radiometric error measure: every extra 6 dB halves the RMSE."
        ),
    },
    "ssim": {
        "label": "SSIM",
        "unit": "",
        "better": "higher",
        "good": 0.80,
        "excellent": 0.90,
        "description": (
            "Gaussian-windowed structural similarity (Wang et al. 2004), averaged over "
            "bands. Compares local luminance, contrast and structure rather than raw "
            "pixel differences, so it tracks perceived sharpness."
        ),
    },
    "rmse": {
        "label": "RMSE",
        "unit": "reflectance",
        "better": "lower",
        "good": 0.03,
        "excellent": 0.015,
        "description": (
            "Root-mean-square reflectance error over all bands, directly comparable to "
            "the Sentinel-2 L2A surface-reflectance uncertainty budget."
        ),
    },
    "mae": {
        "label": "MAE",
        "unit": "reflectance",
        "better": "lower",
        "good": 0.03,
        "excellent": 0.015,
        "description": (
            "Mean absolute reflectance error over all bands. Less sensitive to a few "
            "large outliers than RMSE."
        ),
    },
    "sam_deg": {
        "label": "SAM",
        "unit": "deg",
        "better": "lower",
        "good": 3.0,
        "excellent": 1.5,
        "description": (
            "Spectral Angle Mapper: the mean angle between the predicted and reference "
            "spectra. Invariant to illumination scaling, so it isolates spectral "
            "distortion, the key risk when super-resolving multispectral imagery."
        ),
    },
    "ergas": {
        "label": "ERGAS",
        "unit": "%",
        "better": "lower",
        "good": 6.0,
        "excellent": 3.0,
        "description": (
            "Erreur Relative Globale Adimensionnelle de Synthese: the scale-normalised "
            "relative error aggregated over bands. The standard global quality figure "
            "for pan-sharpening and super-resolution; under 3 is considered excellent."
        ),
    },
    "uiqi": {
        "label": "UIQI",
        "unit": "",
        "better": "higher",
        "good": 0.80,
        "excellent": 0.92,
        "description": (
            "Universal Image Quality Index (Wang & Bovik 2002) on 8x8 blocks: the "
            "product of correlation loss, luminance distortion and contrast distortion."
        ),
    },
    "scc": {
        "label": "SCC",
        "unit": "",
        "better": "higher",
        "good": 0.80,
        "excellent": 0.92,
        "description": (
            "Spatial Correlation Coefficient: correlation of the 3x3 Laplacian high-pass "
            "responses. Measures how much genuine fine spatial detail is recovered, "
            "independently of radiometric bias."
        ),
    },
    "cc": {
        "label": "CC",
        "unit": "",
        "better": "higher",
        "good": 0.80,
        "excellent": 0.92,
        "description": (
            "Pearson correlation coefficient between the reconstruction and the "
            "reference, pooled over all bands and valid pixels."
        ),
    },
}
