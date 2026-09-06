"""Differentiable objectives for spatial detail and spectral fidelity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    pixel: float = 1.0
    spectral: float = 0.25
    gradient: float = 0.1
    source_consistency: float = 0.5
    reflectance_range: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("Loss weights must be non-negative")


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must have shape (batch, bands, height, width)")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} != {target.shape}")
    if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
        raise ValueError("spatial dimensions must be at least 2x2")


def _expand_band_mask(
    band_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Validate a per-band supervision mask and make it broadcastable."""
    if band_mask is None:
        return None

    mask = torch.as_tensor(band_mask, device=reference.device, dtype=reference.dtype)
    if mask.ndim == 1:
        if mask.shape[0] != reference.shape[1]:
            raise ValueError("band_mask must contain one value per target band")
        mask = mask.unsqueeze(0).expand(reference.shape[0], -1)
    elif mask.ndim == 2:
        if mask.shape[1] != reference.shape[1]:
            raise ValueError("band_mask must contain one value per target band")
        if mask.shape[0] == 1 and reference.shape[0] > 1:
            mask = mask.expand(reference.shape[0], -1)
        elif mask.shape[0] != reference.shape[0]:
            raise ValueError("band_mask batch dimension must match prediction")
    else:
        raise ValueError("band_mask must have shape (bands,) or (batch, bands)")

    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("band_mask values must be finite and non-negative")
    if (mask.sum(dim=1) <= 0).any():
        raise ValueError("band_mask must supervise at least one band per sample")
    return mask[:, :, None, None]


def _masked_mean(values: torch.Tensor, band_mask: torch.Tensor | None) -> torch.Tensor:
    if band_mask is None:
        return values.mean()
    expanded = band_mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum()


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Robust pixel reconstruction loss, less sensitive to outliers than MSE."""
    mask = _expand_band_mask(band_mask, prediction)
    residual = torch.sqrt((prediction - target).square() + epsilon**2) - epsilon
    return _masked_mean(residual, mask)


def spectral_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cosine-distance between predicted and target spectral vectors."""
    mask = _expand_band_mask(band_mask, prediction)
    if mask is not None:
        prediction = prediction * mask
        target = target * mask
    pred_vectors = prediction.permute(0, 2, 3, 1)
    target_vectors = target.permute(0, 2, 3, 1)
    cosine = F.cosine_similarity(pred_vectors, target_vectors, dim=-1, eps=epsilon)
    both_zero = (pred_vectors == 0).all(dim=-1) & (target_vectors == 0).all(dim=-1)
    error = (1.0 - cosine).clamp_min(0.0)
    return error.masked_fill(both_zero, 0.0).mean()


def gradient_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 agreement of horizontal and vertical reflectance gradients."""
    mask = _expand_band_mask(band_mask, prediction)
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return _masked_mean((pred_dx - target_dx).abs(), mask) + _masked_mean(
        (pred_dy - target_dy).abs(), mask
    )


def source_consistency_loss(prediction: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Require area-downsampled SR reflectance to reproduce its LR observation."""
    if source.ndim != 4:
        raise ValueError("source must have shape (batch, bands, height, width)")
    if prediction.shape[:2] != source.shape[:2]:
        raise ValueError("prediction and source batch/band dimensions must match")
    if prediction.shape[-2] < source.shape[-2] or prediction.shape[-1] < source.shape[-1]:
        raise ValueError("prediction must be at least as large as source spatially")
    downsampled = F.interpolate(prediction, size=source.shape[-2:], mode="area")
    return F.l1_loss(downsampled, source)


def reflectance_range_loss(
    prediction: torch.Tensor,
    minimum: float = 0.0,
    maximum: float = 1.5,
) -> torch.Tensor:
    """Penalize reflectance outside the physically permitted working range."""
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    below = F.relu(minimum - prediction)
    above = F.relu(prediction - maximum)
    return (below + above).mean()


class SpectralSpatialLoss(nn.Module):
    """Composite paired-training objective for Sentinel-2 super-resolution.

    The objective combines robust reconstruction, spectral-vector agreement,
    boundary preservation, consistency with the observed LR image, and a soft
    physical reflectance constraint. Call :meth:`components` to log each term.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        reflectance_min: float = 0.0,
        reflectance_max: float = 1.5,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.reflectance_min = reflectance_min
        self.reflectance_max = reflectance_max

    def components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor | None = None,
        band_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_pair(prediction, target)
        terms = {
            "pixel": charbonnier_loss(prediction, target, band_mask=band_mask),
            "spectral": spectral_consistency_loss(prediction, target, band_mask=band_mask),
            "gradient": gradient_consistency_loss(prediction, target, band_mask=band_mask),
            "reflectance_range": reflectance_range_loss(
                prediction, self.reflectance_min, self.reflectance_max
            ),
        }
        if source is not None:
            terms["source_consistency"] = source_consistency_loss(prediction, source)
        else:
            terms["source_consistency"] = prediction.new_zeros(())

        total = (
            self.weights.pixel * terms["pixel"]
            + self.weights.spectral * terms["spectral"]
            + self.weights.gradient * terms["gradient"]
            + self.weights.source_consistency * terms["source_consistency"]
            + self.weights.reflectance_range * terms["reflectance_range"]
        )
        return {"total": total, **terms}

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor | None = None,
        band_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.components(prediction, target, source, band_mask)["total"]
