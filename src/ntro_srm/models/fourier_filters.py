"""Vectorized Fourier low-pass filters and cached inference helpers.

Replaces the explicit Python-loop filter construction in
``third_party/SEN2SR/sen2sr/models/tricks.py`` (``ideal_filter``,
``butterworth_filter``, ``gaussian_filter``, ``sigmoid_filter``) with fully
vectorized PyTorch equivalents, plus a small device-aware cache and an
efficient hard-constraint forward.

The upstream repository is intentionally left untouched (SIH rule: vendor code
is read-only).  Everything here lives in our own package and is applied at
runtime by :func:`optimize_upstream_masks`.

Design notes
------------
* Builders accept arbitrary rectangular ``(h, w)`` shapes (upstream loops
  already did, but the tiled inference around them assumed squares).
* Numerics match the loop reference exactly in float32 (verified in
  ``tests/test_inference_opt.py``).
* :func:`get_low_pass_mask` caches the CPU master copy; per-device views are
  cached separately so repeated inferences never rebuild masks.
* :func:`apply_hard_constraint` uses ``fft2``/``ifft2`` with
  ``dim=(-2, -1)`` (upstream shifts *all* dims, including batch/channel,
  which is a no-op mathematically but wastes time), precomputed
  complementary high-pass masks, and no per-call Python loops.
"""

from __future__ import annotations

import math
import types
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

FilterMethod = Literal["ideal", "butterworth", "sigmoid", "gaussian"]

_MASK_CACHE: Dict[tuple, torch.Tensor] = {}
_CACHE_ORDER: list[tuple] = []
_CACHE_MAX = 64


def _freq_distances(
    h: int,
    w: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (dist, dist_sq, center_h, center_w) frequency grids."""
    crow, ccol = h // 2, w // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    dy = yy - float(crow)
    dx = xx - float(ccol)
    dist_sq = dy * dy + dx * dx
    return torch.sqrt(dist_sq), dist_sq, crow, ccol


def ideal_filter(
    shape: Tuple[int, int],
    cutoff: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorized ideal (binary disk) low-pass filter."""
    h, w = int(shape[0]), int(shape[1])
    dist, _, _, _ = _freq_distances(h, w, device=device, dtype=torch.float32)
    return (dist <= float(cutoff)).to(dtype)


def butterworth_filter(
    shape: Tuple[int, int],
    cutoff: int,
    order: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorized Butterworth low-pass filter."""
    h, w = int(shape[0]), int(shape[1])
    cutoff_f = float(cutoff) if float(cutoff) != 0.0 else 1e-6
    dist, _, _, _ = _freq_distances(h, w, device=device, dtype=torch.float32)
    out = 1.0 / (1.0 + (dist / cutoff_f) ** (2 * int(order)))
    return out.to(dtype)


def gaussian_filter(
    shape: Tuple[int, int],
    cutoff: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorized Gaussian low-pass filter."""
    h, w = int(shape[0]), int(shape[1])
    cutoff_f = float(cutoff) if float(cutoff) != 0.0 else 1e-6
    _, dist_sq, _, _ = _freq_distances(h, w, device=device, dtype=torch.float32)
    out = torch.exp(-dist_sq / (2.0 * cutoff_f * cutoff_f))
    return out.to(dtype)


def sigmoid_filter(
    shape: Tuple[int, int],
    cutoff: int,
    sharpness: float,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorized sigmoid low-pass filter."""
    h, w = int(shape[0]), int(shape[1])
    sharp = float(sharpness) if float(sharpness) != 0.0 else 1e-6
    dist, _, _, _ = _freq_distances(h, w, device=device, dtype=torch.float32)
    out = 1.0 / (1.0 + torch.exp((dist - float(cutoff)) / sharp))
    return out.to(dtype)


def _cache_key(
    method: str,
    h: int,
    w: int,
    cutoff: float,
    order: int,
    sharpness: float,
    device_type: str,
    dtype_str: str,
) -> tuple:
    return (method, int(h), int(w), float(cutoff), int(order), float(sharpness), device_type, dtype_str)


def get_low_pass_mask(
    filter_method: FilterMethod,
    shape: Tuple[int, int],
    cutoff: Union[int, float],
    order: int = 2,
    sharpness: float = 10.0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a cached vectorized low-pass mask for the requested spec.

    The cache is keyed by (method, h, w, cutoff, order, sharpness, device,
    dtype) and bounded to ``_CACHE_MAX`` entries with FIFO eviction.
    """
    dev = torch.device(device)
    key = _cache_key(
        str(filter_method), int(shape[0]), int(shape[1]),
        float(cutoff), int(order), float(sharpness), dev.type, str(dtype),
    )
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit
    method = str(filter_method).lower()
    h, w = int(shape[0]), int(shape[1])
    if method == "ideal":
        mask = ideal_filter((h, w), int(round(float(cutoff))), device=dev, dtype=dtype)
    elif method == "butterworth":
        mask = butterworth_filter((h, w), int(round(float(cutoff))), order=int(order), device=dev, dtype=dtype)
    elif method == "gaussian":
        mask = gaussian_filter((h, w), int(round(float(cutoff))), device=dev, dtype=dtype)
    elif method == "sigmoid":
        mask = sigmoid_filter((h, w), int(round(float(cutoff))), sharpness=float(sharpness), device=dev, dtype=dtype)
    else:
        raise ValueError(f"Unsupported filter_method '{filter_method}'")
    mask = mask.contiguous()
    _MASK_CACHE[key] = mask
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _MASK_CACHE.pop(old, None)
    return mask


def build_mask_for_scale(
    filter_method: FilterMethod,
    sr_shape: Tuple[int, int],
    scale_factor: int,
    filter_hyperparameters: Optional[dict] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Mirror upstream radius convention: ``min(ch, cw) // scale``."""
    h, w = int(sr_shape[0]), int(sr_shape[1])
    radius = min(h // 2, w // 2) // int(scale_factor)
    params = filter_hyperparameters or {}
    return get_low_pass_mask(
        filter_method, (h, w), radius,
        order=int(params.get("order", 2)),
        sharpness=float(params.get("sharpness", 10.0)),
        device=device, dtype=dtype,
    )


def clear_mask_cache() -> None:
    """Empty the Fourier mask cache (useful for memory accounting tests)."""
    _MASK_CACHE.clear()
    _CACHE_ORDER.clear()


def cache_stats() -> dict:
    """Return cache occupancy for logging/debugging."""
    return {"entries": len(_MASK_CACHE), "max_entries": _CACHE_MAX}


def apply_hard_constraint(
    lr: torch.Tensor,
    sr: torch.Tensor,
    low_pass_mask: torch.Tensor,
    high_pass_mask: Optional[torch.Tensor] = None,
    bands: Union[str, list, tuple] = "all",
) -> torch.Tensor:
    """Efficient Fourier hard-constraint forward.

    Mathematically equivalent to upstream ``HardConstraint.forward`` /
    ``FourierHardConstraint.forward`` but:

    * uses ``fft2``/``ifft2`` with ``dim=(-2, -1)`` instead of ``fftn`` plus
      full-tensor ``fftshift`` (upstream rolls batch/channel dims too, which
      commutes with the broadcast multiply but wastes time);
    * reuses a precomputed ``high_pass_mask`` instead of ``1 - mask`` per call;
    * never rebuilds masks inside the hot loop.
    """
    if bands != "all":
        lr_use = lr[:, list(bands)]
    else:
        lr_use = lr
    lr_up = F.interpolate(lr_use, size=sr.shape[-2:], mode="bicubic", antialias=True)

    low = low_pass_mask.to(dtype=torch.float32, device=sr.device, non_blocking=True)
    if high_pass_mask is None:
        high = (1.0 - low).contiguous()
    else:
        high = high_pass_mask.to(dtype=torch.float32, device=sr.device, non_blocking=True)

    sr_fft = torch.fft.fft2(sr.float(), dim=(-2, -1))
    lr_fft = torch.fft.fft2(lr_up.float(), dim=(-2, -1))
    sr_shift = torch.fft.fftshift(sr_fft, dim=(-2, -1))
    lr_shift = torch.fft.fftshift(lr_fft, dim=(-2, -1))

    combined = lr_shift * low + sr_shift * high
    out = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1))
    result = torch.real(out)
    return result.to(sr.dtype) if result.dtype != sr.dtype else result


def _optimized_forward(self, lr: torch.Tensor, sr: torch.Tensor) -> torch.Tensor:
    low = getattr(self, "low_pass_mask", None)
    if not isinstance(low, torch.Tensor):
        raise RuntimeError("Optimized Fourier module is missing 'low_pass_mask'")
    high = getattr(self, "_opt_high_pass_mask", None)
    bands = getattr(self, "bands", "all")
    return apply_hard_constraint(lr, sr, low, high, bands=bands)


def optimize_upstream_masks(model: torch.nn.Module, device: Union[str, torch.device]) -> int:
    """Move upstream Fourier masks to ``device`` and patch efficient forwards.

    * Makes each ``low_pass_mask`` contiguous float32 on the target device.
    * Precomputes and caches the complementary high-pass mask as
      ``module._opt_high_pass_mask`` (plain tensor attr, so upstream
      checkpoints/state-dicts are unaffected).
    * Replaces the per-instance ``forward`` with the vectorized FFT path.
      The operation is idempotent (tracked via ``_opt_patched``).

    Returns the number of patched modules.
    """
    dev = torch.device(device)
    patched = 0
    seen_shapes: Dict[tuple, torch.Tensor] = {}
    for module in model.modules():
        low = getattr(module, "low_pass_mask", None)
        if not isinstance(low, torch.Tensor):
            continue
        key = tuple(low.shape)
        canonical = seen_shapes.get(key)
        if canonical is not None and canonical.shape == low.shape:
            try:
                if torch.equal(canonical.cpu(), low.detach().cpu()):
                    low = canonical
            except Exception:
                pass
        low = low.detach().to(dtype=torch.float32, device=dev).contiguous()
        module.low_pass_mask = low
        module._opt_high_pass_mask = (1.0 - low).contiguous()
        seen_shapes.setdefault(key, low)
        if not getattr(module, "_opt_patched", False):
            module.forward = types.MethodType(_optimized_forward, module)  # type: ignore[method-assign]
            module._opt_patched = True  # type: ignore[attr-defined]
        patched += 1
    return patched
