"""Tests for the spectrally constrained training objective."""

import pytest
import torch

from ntro_srm.training.losses import LossWeights, SpectralSpatialLoss


def test_identical_prediction_has_near_zero_loss() -> None:
    target = torch.rand(2, 10, 16, 16)
    source = torch.nn.functional.interpolate(target, size=(4, 4), mode="area")
    objective = SpectralSpatialLoss()
    components = objective.components(target.clone(), target, source)

    assert components["total"].item() < 1e-6
    assert components["spectral"].item() < 1e-6
    assert components["gradient"].item() == 0.0
    assert components["source_consistency"].item() == 0.0


def test_distortion_produces_loss_and_finite_gradients() -> None:
    target = torch.rand(1, 4, 16, 16)
    source = torch.nn.functional.interpolate(target, size=(4, 4), mode="area")
    prediction = (target + torch.randn_like(target) * 0.05).requires_grad_(True)
    objective = SpectralSpatialLoss()

    loss = objective(prediction, target, source)
    loss.backward()

    assert loss.item() > 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_invalid_weights_and_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LossWeights(spectral=-1.0)

    objective = SpectralSpatialLoss()
    with pytest.raises(ValueError, match="shapes differ"):
        objective(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 4, 4))
