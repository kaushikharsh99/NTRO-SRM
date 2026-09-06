"""Spatial and spectral metrics for multispectral super-resolution."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _validate_arrays(
    prediction: np.ndarray,
    reference: np.ndarray,
    band_names: Sequence[str] | None,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.ndim != 3 or ref.ndim != 3:
        raise ValueError("prediction and reference must have shape (bands, height, width)")
    if pred.shape != ref.shape:
        raise ValueError(f"prediction and reference shapes differ: {pred.shape} != {ref.shape}")

    names = list(band_names) if band_names is not None else [f"band_{i + 1}" for i in range(pred.shape[0])]
    if len(names) != pred.shape[0]:
        raise ValueError(f"Expected {pred.shape[0]} band names, received {len(names)}")

    finite = np.isfinite(pred).all(axis=0) & np.isfinite(ref).all(axis=0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != pred.shape[1:]:
            raise ValueError(f"valid_mask shape {supplied.shape} does not match {pred.shape[1:]}")
        finite &= supplied
    if not finite.any():
        raise ValueError("No valid pixels remain after applying nodata and finite-value masks")
    return pred, ref, names, finite


def _gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    weights = torch.exp(-(coords**2) / (2 * sigma**2))
    weights /= weights.sum()
    return torch.outer(weights, weights).reshape(1, 1, size, size)


def _ssim_per_band(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    data_range: float,
) -> np.ndarray:
    """Compute Gaussian-window SSIM independently for every band."""
    height, width = prediction.shape[-2:]
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)

    x = torch.from_numpy(prediction).unsqueeze(1)
    y = torch.from_numpy(reference).unsqueeze(1)
    window = _gaussian_window(window_size)
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding)
    mu_y = F.conv2d(y, window, padding=padding)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x.square(), window, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(y.square(), window, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = (numerator / denominator.clamp_min(1e-15)).squeeze(1).numpy()
    return np.asarray([float(values[valid_mask].mean()) for values in ssim_map])


def spectral_angle_mapper(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Return mean spectral angle in degrees across valid, non-dark pixels."""
    pred_vectors = prediction[:, valid_mask]
    ref_vectors = reference[:, valid_mask]
    pred_norm = np.linalg.norm(pred_vectors, axis=0)
    ref_norm = np.linalg.norm(ref_vectors, axis=0)
    usable = (pred_norm > epsilon) & (ref_norm > epsilon)
    if not usable.any():
        return 0.0
    cosine = np.sum(pred_vectors[:, usable] * ref_vectors[:, usable], axis=0)
    cosine /= pred_norm[usable] * ref_norm[usable]
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(angles).mean())


def _index_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    band_names: Sequence[str],
    numerator_band: str,
    denominator_band: str,
) -> dict[str, float] | None:
    if numerator_band not in band_names or denominator_band not in band_names:
        return None
    numerator_idx = band_names.index(numerator_band)
    denominator_idx = band_names.index(denominator_band)

    def normalized_difference(array: np.ndarray) -> np.ndarray:
        a = array[numerator_idx]
        b = array[denominator_idx]
        return (a - b) / np.maximum(np.abs(a + b), 1e-8)

    difference = normalized_difference(prediction)[valid_mask] - normalized_difference(reference)[valid_mask]
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "bias": float(np.mean(difference)),
    }


def evaluate_arrays(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    band_names: Sequence[str] | None = None,
    valid_mask: np.ndarray | None = None,
    data_range: float = 1.0,
    scale_factor: float = 4.0,
) -> dict:
    """Evaluate aligned reflectance arrays using spatial and spectral metrics."""
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    pred, ref, names, mask = _validate_arrays(prediction, reference, band_names, valid_mask)
    difference = pred - ref
    per_band: dict[str, dict[str, float]] = {}
    relative_squared_errors: list[float] = []
    ssim_values = _ssim_per_band(pred, ref, mask, data_range)

    for index, name in enumerate(names):
        delta = difference[index][mask]
        reference_values = ref[index][mask]
        mse = float(np.mean(delta**2))
        rmse = float(np.sqrt(mse))
        mean_reference = float(np.mean(reference_values))
        if abs(mean_reference) > 1e-8:
            relative_squared_errors.append((rmse / mean_reference) ** 2)
        per_band[name] = {
            "mae": float(np.mean(np.abs(delta))),
            "rmse": rmse,
            "bias": float(np.mean(delta)),
            "psnr_db": float(10.0 * np.log10((data_range**2) / max(mse, 1e-12))),
            "ssim": float(ssim_values[index]),
        }

    aggregate = {
        "mae": float(np.mean(np.abs(difference[:, mask]))),
        "rmse": float(np.sqrt(np.mean(difference[:, mask] ** 2))),
        "mean_psnr_db": float(np.mean([metrics["psnr_db"] for metrics in per_band.values()])),
        "mean_ssim": float(np.mean(ssim_values)),
        "sam_degrees": spectral_angle_mapper(pred, ref, mask),
        "ergas": (
            float(100.0 / scale_factor * np.sqrt(np.mean(relative_squared_errors)))
            if relative_squared_errors
            else 0.0
        ),
        "valid_pixels": int(mask.sum()),
        "valid_fraction": float(mask.mean()),
    }

    spectral_indices: dict[str, dict[str, float]] = {}
    ndvi = _index_metrics(pred, ref, mask, names, "B08", "B04")
    if ndvi is not None:
        spectral_indices["ndvi"] = ndvi
    ndwi = _index_metrics(pred, ref, mask, names, "B03", "B08")
    if ndwi is not None:
        spectral_indices["ndwi"] = ndwi

    return {"aggregate": aggregate, "per_band": per_band, "spectral_indices": spectral_indices}
