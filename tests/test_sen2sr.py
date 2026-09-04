"""Unit tests for NTRO-SRM SEN2SR adapter and preprocessing transforms."""

from pathlib import Path
import pytest
import torch

from ntro_srm.models.sen2sr import SEN2SRModel
from ntro_srm.preprocessing.transforms import (
    S2_10BAND_NAMES,
    clamp_non_negative,
    denormalize_reflectance,
    handle_nans,
    normalize_reflectance,
    rgbn_to_s2_10band,
    s2_10band_to_rgbn,
)


@pytest.fixture(scope="module")
def sen2sr_model() -> SEN2SRModel:
    """Fixture providing a cached SEN2SR-Lite model instance."""
    ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "SEN2SRLite"
    return SEN2SRModel(
        model_variant="lite",
        device="cpu",
        checkpoint_dir=ckpt_dir,
        auto_download=True,
    )


class TestPreprocessingTransforms:
    """Tests for basic radiometric and band-handling transformations."""

    def test_normalization(self):
        raw = torch.tensor([0.0, 5000.0, 10000.0, 15000.0])
        norm = normalize_reflectance(raw, scale=10000.0)
        assert torch.allclose(norm, torch.tensor([0.0, 0.5, 1.0, 1.5]))

        denorm = denormalize_reflectance(norm, scale=10000.0, dtype=torch.float32)
        assert torch.allclose(denorm[:3], raw[:3])

    def test_handle_nans(self):
        t = torch.tensor([1.0, float("nan"), float("inf"), float("-inf"), 0.5])
        clean = handle_nans(t, fill_value=0.0)
        assert torch.isfinite(clean).all()
        assert clean[1].item() == 0.0
        assert clean[2].item() == 0.0
        assert clean[3].item() == 0.0
        assert clean[0].item() == 1.0

    def test_clamp_non_negative(self):
        t = torch.tensor([-0.5, -0.01, 0.0, 0.3, 1.2])
        clamped = clamp_non_negative(t, min_value=0.0)
        assert (clamped >= 0.0).all()
        assert clamped[0].item() == 0.0
        assert clamped[3].item() == pytest.approx(0.3, abs=1e-5)

    def test_band_order_conversion(self):
        # Shape (1, 10, 8, 8) with known band index values
        s2_stack = torch.zeros(1, 10, 8, 8)
        for i in range(10):
            s2_stack[:, i, :, :] = float(i)

        # S2 to RGBN [B04=2, B03=1, B02=0, B08=6]
        rgbn = s2_10band_to_rgbn(s2_stack)
        assert rgbn.shape == (1, 4, 8, 8)
        assert rgbn[0, 0, 0, 0].item() == 2.0  # Red (B04)
        assert rgbn[0, 1, 0, 0].item() == 1.0  # Green (B03)
        assert rgbn[0, 2, 0, 0].item() == 0.0  # Blue (B02)
        assert rgbn[0, 3, 0, 0].item() == 6.0  # NIR (B08)

        # Reconstruct with dummy RSWIR
        rswir = torch.zeros(1, 6, 8, 8)
        for idx, val in enumerate([3, 4, 5, 7, 8, 9]):
            rswir[:, idx, :, :] = float(val)

        reconstructed = rgbn_to_s2_10band(rgbn, rswir)
        assert reconstructed.shape == (1, 10, 8, 8)
        assert torch.allclose(reconstructed, s2_stack)


class TestSEN2SRAdapter:
    """Tests for the SEN2SRModel adapter."""

    def test_output_shape_and_4x_scale_native(self, sen2sr_model: SEN2SRModel):
        in_h, in_w = 128, 128
        x = torch.rand(1, 10, in_h, in_w, dtype=torch.float32) * 0.5
        sr = sen2sr_model.predict(x, auto_normalize=False)

        assert sr.shape == (1, 10, in_h * 4, in_w * 4)
        assert sr.dtype == torch.float32

    def test_output_shape_and_4x_scale_padded(self, sen2sr_model: SEN2SRModel):
        in_h, in_w = 64, 64
        x = torch.rand(1, 10, in_h, in_w, dtype=torch.float32) * 0.5
        sr = sen2sr_model.predict(x, auto_normalize=False)

        assert sr.shape == (1, 10, in_h * 4, in_w * 4)
        assert sr.dtype == torch.float32

    def test_output_shape_non_square(self, sen2sr_model: SEN2SRModel):
        # Non-square inputs: odd dimensions (126x127) and larger non-square (130x140)
        for h, w in [(126, 127), (130, 140), (95, 110)]:
            x = torch.rand(1, 10, h, w, dtype=torch.float32) * 0.5
            sr = sen2sr_model.predict(x, auto_normalize=False)
            assert sr.shape == (1, 10, h * 4, w * 4), f"Failed for shape ({h}, {w})"
            assert sr.dtype == torch.float32


    def test_3d_input_support(self, sen2sr_model: SEN2SRModel):
        in_h, in_w = 128, 128
        x = torch.rand(10, in_h, in_w, dtype=torch.float32) * 0.5
        sr = sen2sr_model.predict(x, auto_normalize=False)

        assert sr.shape == (10, in_h * 4, in_w * 4)

    def test_expected_band_count_validation(self, sen2sr_model: SEN2SRModel):
        # 4 bands instead of 10 should be rejected
        bad_tensor = torch.rand(1, 4, 128, 128)
        with pytest.raises(ValueError, match="Expected 10 Sentinel-2 bands"):
            sen2sr_model.predict(bad_tensor)

    def test_auto_normalization(self, sen2sr_model: SEN2SRModel):
        # Raw Sentinel-2 L2A integers [0, 10000]
        raw_s2 = torch.randint(100, 4000, (1, 10, 128, 128), dtype=torch.float32)
        sr = sen2sr_model.predict(raw_s2, auto_normalize=True, clamp_output=True)

        assert sr.shape == (1, 10, 512, 512)
        assert sr.min() >= 0.0
        assert sr.max() <= 1.5  # Realistic normalized reflectance

    def test_nan_handling(self, sen2sr_model: SEN2SRModel):
        x = torch.rand(1, 10, 128, 128, dtype=torch.float32) * 0.4
        x[:, :, 10:15, 10:15] = float("nan")
        x[:, :, 20:25, 20:25] = float("inf")

        sr = sen2sr_model.predict(x, auto_normalize=False, clamp_output=True)
        assert torch.isfinite(sr).all(), "Output must not contain NaNs or Infs"

    def test_output_validity(self, sen2sr_model: SEN2SRModel):
        x = torch.rand(1, 10, 128, 128, dtype=torch.float32) * 0.3 + 0.05
        sr = sen2sr_model.predict(x, auto_normalize=False, clamp_output=True)

        assert bool(torch.isfinite(sr).all().item())
        assert (sr >= 0.0).all(), "Surface reflectance must be non-negative"
        assert sr.mean() > 0.0
