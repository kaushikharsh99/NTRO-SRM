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


def test_zero_prediction_target_and_source_have_zero_loss() -> None:
    prediction = torch.zeros(1, 10, 8, 8)
    target = torch.zeros_like(prediction)
    source = torch.zeros(1, 10, 2, 2)

    components = SpectralSpatialLoss().components(prediction, target, source)

    assert components["spectral"].item() == 0.0
    assert components["total"].item() == 0.0


def test_invalid_weights_and_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LossWeights(spectral=-1.0)

    objective = SpectralSpatialLoss()
    with pytest.raises(ValueError, match="shapes differ"):
        objective(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 4, 4))


def test_band_mask_ignores_unmeasured_reference_channels() -> None:
    target = torch.full((1, 4, 8, 8), 0.25)
    prediction = target.clone()
    prediction[:, 2:] = 1.25
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    objective = SpectralSpatialLoss(
        LossWeights(source_consistency=0.0, reflectance_range=0.0)
    )

    components = objective.components(prediction, target, band_mask=mask)

    assert components["total"].item() < 1e-6
    assert components["pixel"].item() == 0.0
    assert components["spectral"].item() < 1e-6
    assert components["gradient"].item() == 0.0


def test_band_mask_blocks_reference_gradients_for_unmeasured_channels() -> None:
    target = torch.rand(1, 4, 8, 8)
    prediction = torch.rand(1, 4, 8, 8, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    objective = SpectralSpatialLoss(
        LossWeights(source_consistency=0.0, reflectance_range=0.0)
    )

    objective(prediction, target, band_mask=mask).backward()

    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[:, :2]) > 0
    assert torch.count_nonzero(prediction.grad[:, 2:]) == 0


@pytest.mark.parametrize(
    "mask, message",
    [
        (torch.ones(3), "one value per target band"),
        (torch.ones(2, 4), "batch dimension"),
        (torch.zeros(4), "at least one band"),
    ],
)
def test_invalid_band_masks_are_rejected(mask: torch.Tensor, message: str) -> None:
    objective = SpectralSpatialLoss()
    prediction = torch.rand(1, 4, 8, 8)

    with pytest.raises(ValueError, match=message):
        objective(prediction, prediction, band_mask=mask)
