"""Shared array helpers for the NTRO-SRM evaluation package.

Every public evaluation function accepts either a NumPy array or a PyTorch tensor. These
helpers normalise that input to a canonical channel-first NumPy view and provide the
NaN-tolerant masking and summary statistics the metric, index and uncertainty modules all
need, so that behaviour on degenerate rasters is identical across the package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_STAT_KEYS: tuple[str, ...] = (
    "mean", "std", "min", "max", "p01", "p05", "p25", "p50", "p75", "p95", "p99",
)


def to_chw_numpy(x: Any, dtype: Any = np.float32) -> np.ndarray:
    """Convert a tensor or array to a contiguous ``(C, H, W)`` floating-point NumPy array.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input of shape ``(H, W)``, ``(C, H, W)`` or ``(1, C, H, W)``. PyTorch tensors are
        detached and moved to the CPU first.
    dtype : numpy dtype, default=numpy.float32
        Floating-point dtype of the returned array.

    Returns
    -------
    numpy.ndarray
        Contiguous ``(C, H, W)`` array. A 2-D input gains a leading singleton channel.

    Raises
    ------
    ValueError
        If the input has fewer than two or more than four dimensions, or if a 4-D input
        carries a batch dimension larger than one.
    """
    if hasattr(x, "detach"):  # torch.Tensor without importing torch here
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a single sample, got a batch of {arr.shape[0]} with shape {arr.shape}."
            )
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected a 2-D, 3-D or 4-D array, got shape {arr.shape}.")

    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """Build the ``(H, W)`` mask of pixels finite across every band of every input.

    Parameters
    ----------
    *arrays : numpy.ndarray
        One or more ``(C, H, W)`` or ``(H, W)`` arrays sharing the same spatial shape.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` mask. Returns an all-``True`` mask when no arrays are supplied.

    Raises
    ------
    ValueError
        If the inputs do not share the same spatial dimensions.
    """
    if not arrays:
        return np.ones((0, 0), dtype=bool)

    mask: np.ndarray | None = None
    for arr in arrays:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = a[np.newaxis, ...]
        finite = np.isfinite(a).all(axis=0)
        if mask is None:
            mask = finite
        else:
            if mask.shape != finite.shape:
                raise ValueError(
                    f"Spatial shapes differ: {mask.shape} vs {finite.shape}."
                )
            mask = mask & finite
    return mask


def safe_stats(arr: np.ndarray) -> dict[str, float]:
    """Summarise the finite values of an array, tolerating fully invalid input.

    Parameters
    ----------
    arr : numpy.ndarray
        Array of any shape.

    Returns
    -------
    dict[str, float]
        Keys ``mean``, ``std``, ``min``, ``max`` and the percentiles ``p01``, ``p05``,
        ``p25``, ``p50``, ``p75``, ``p95``, ``p99``. Every value is ``nan`` when the input
        contains no finite element.
    """
    values = np.asarray(arr, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in _STAT_KEYS}

    percentiles = np.percentile(values, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }
