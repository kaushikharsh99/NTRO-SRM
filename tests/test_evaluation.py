"""Unit tests for the NTRO-SRM evaluation package.

Covers the full-reference metrics, the hand-rolled colour lookup tables, Wald's
synthesis and consistency protocols, the uncertainty/hallucination analysis, the
spectral index registry and the quality report. Every test runs on the CPU against
deterministic synthetic rasters with a bicubic upsampler standing in for the
super-resolution network: no checkpoint, no GPU and no network access are required.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ntro_srm.evaluation import (
    colormaps,
    consistency,
    indices,
    metrics,
    report,
    uncertainty,
)
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# Canonical 10-band reflectance signatures (B02 B03 B04 B05 B06 B07 B08 B8A B11 B12).
VEGETATION_SPECTRUM = [0.03, 0.06, 0.04, 0.10, 0.28, 0.34, 0.38, 0.40, 0.20, 0.09]
WATER_SPECTRUM = [0.09, 0.11, 0.07, 0.05, 0.03, 0.02, 0.015, 0.012, 0.008, 0.005]


def _synthetic_stack(bands: int = 10, size: int = 32, seed: int = 7) -> torch.Tensor:
    """Deterministic, smoothly varying (C, H, W) reflectance stack in [0, 1]."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(bands, size // 4, size // 4, generator=generator)
    smooth = F.interpolate(
        coarse.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False
    )[0]
    rows, cols = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    texture = 0.08 * torch.sin(3.0 * cols) * torch.cos(2.5 * rows)
    return (0.4 * smooth + 0.1 + texture.unsqueeze(0)).clamp(0.02, 0.95).float()


def _spectrum_stack(spectrum: list[float], size: int = 16) -> torch.Tensor:
    """(C, size, size) stack in which every pixel carries the same spectrum."""
    values = torch.tensor(spectrum, dtype=torch.float32).view(-1, 1, 1)
    return values.expand(len(spectrum), size, size).contiguous()


def bicubic_predict(tensor: torch.Tensor) -> torch.Tensor:
    """Stand-in super-resolution model: a plain 4x bicubic upsample."""
    return consistency.upsample_bicubic(tensor, factor=4)


def nearest_predict(tensor: torch.Tensor) -> torch.Tensor:
    """Deliberately poor model: blocky 4x nearest-neighbour replication."""
    return F.interpolate(tensor.unsqueeze(0), scale_factor=4, mode="nearest")[0]


@pytest.fixture(scope="module")
def lr_stack() -> torch.Tensor:
    """Observed 10-band 10 m stack of shape (10, 32, 32)."""
    return _synthetic_stack()


@pytest.fixture(scope="module")
def sr_stack(lr_stack: torch.Tensor) -> torch.Tensor:
    """Bicubic 4x super-resolved product of shape (10, 128, 128)."""
    return consistency.upsample_bicubic(lr_stack, factor=4)


@pytest.fixture(scope="module")
def wald_result(lr_stack: torch.Tensor) -> consistency.WaldValidationResult:
    """Wald validation of the bicubic stand-in model."""
    return consistency.wald_protocol_validate(lr_stack, bicubic_predict, factor=4)


class TestMetrics:
    """Full-reference image quality metrics."""

    def test_identity_is_perfect(self, lr_stack: torch.Tensor):
        result = metrics.compute_all(lr_stack, lr_stack)
        assert result["psnr_db"] > 90.0, "identical stacks must score a near-infinite PSNR"
        assert result["ssim"] == pytest.approx(1.0, abs=1e-6), "identity SSIM must be 1"
        assert result["sam_deg"] == pytest.approx(0.0, abs=1e-4), "identity SAM must be 0 deg"
        assert result["uiqi"] == pytest.approx(1.0, abs=1e-6), "identity UIQI must be 1"
        assert result["scc"] == pytest.approx(1.0, abs=1e-6), "identity SCC must be 1"
        assert result["cc"] == pytest.approx(1.0, abs=1e-6), "identity CC must be 1"
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9), "identity RMSE must be 0"

    def test_metrics_are_monotonic_in_noise(self, lr_stack: torch.Tensor):
        reference = lr_stack.numpy()
        noise = np.random.default_rng(0).normal(0.0, 1.0, reference.shape).astype(np.float32)
        mild = metrics.compute_all(reference + 0.01 * noise, reference)
        harsh = metrics.compute_all(reference + 0.06 * noise, reference)

        for key in ("psnr_db", "ssim", "uiqi", "scc", "cc"):
            assert mild[key] > harsh[key], f"{key} must decrease as noise grows"
        for key in ("rmse", "mae", "sam_deg", "ergas"):
            assert mild[key] < harsh[key], f"{key} must increase as noise grows"

    def test_compute_all_key_set_and_json(self, lr_stack: torch.Tensor):
        result = metrics.compute_all(lr_stack + 0.01, lr_stack)
        assert set(result) == {"psnr_db", "ssim", "rmse", "mae", "sam_deg", "ergas",
                               "uiqi", "scc", "cc", "n_bands", "shape", "per_band"}, (
            "compute_all must return exactly the contracted key set")
        assert result["n_bands"] == 10
        assert result["shape"] == [10, 32, 32], "shape must be reported as [C, H, W]"
        json.dumps(result, allow_nan=False)

    def test_per_band_table(self, lr_stack: torch.Tensor):
        result = metrics.compute_all(lr_stack + 0.02, lr_stack)
        assert len(result["per_band"]) == 10, "one per_band entry per band"
        assert [row["band"] for row in result["per_band"]] == S2_10BAND_NAMES
        for row in result["per_band"]:
            assert set(row) == {"band", "psnr_db", "ssim", "rmse", "cc"}

        three = metrics.compute_all(lr_stack[:3] + 0.02, lr_stack[:3])
        assert [row["band"] for row in three["per_band"]] == ["Band 1", "Band 2", "Band 3"]

    def test_nan_tolerance(self, lr_stack: torch.Tensor):
        pred = lr_stack.clone()
        ref = lr_stack.clone()
        pred[:, 4:8, 4:8] = float("nan")
        ref[:, 20:24, :] = float("nan")
        partial = metrics.compute_all(pred, ref)
        assert math.isfinite(partial["psnr_db"]), "NaN pixels must be masked, not propagated"
        assert math.isfinite(partial["rmse"])

        degenerate = metrics.compute_all(torch.full_like(lr_stack, float("nan")), lr_stack)
        assert math.isnan(degenerate["rmse"]), "an all-NaN pair is genuinely undefined"
        json.dumps(degenerate)

    def test_metric_meta_covers_every_scalar_metric(self, lr_stack: torch.Tensor):
        scalars = set(metrics.compute_all(lr_stack + 0.01, lr_stack))
        scalars -= {"n_bands", "shape", "per_band"}
        assert set(metrics.METRIC_META) == scalars, "METRIC_META must describe every scalar"

        for key, meta in metrics.METRIC_META.items():
            missing = {"label", "unit", "better", "good", "excellent", "description"} - set(meta)
            assert not missing, f"METRIC_META[{key!r}] is missing {sorted(missing)}"
            assert meta["better"] in {"higher", "lower"}, f"{key} has an invalid polarity"
            if meta["better"] == "lower":
                assert meta["good"] > meta["excellent"], f"{key} thresholds are inverted"
            else:
                assert meta["excellent"] > meta["good"], f"{key} thresholds are inverted"
            assert meta["description"].strip(), f"{key} needs a description"


class TestColormaps:
    """Hand-rolled colour lookup tables and array-to-RGB rendering."""

    def test_every_registered_lut(self):
        assert colormaps.COLORMAP_STOPS, "the colormap registry must not be empty"
        for name in colormaps.COLORMAP_STOPS:
            lut = colormaps.get_colormap(name)
            assert lut.shape == (256, 3), f"{name} LUT must be (256, 3), got {lut.shape}"
            assert lut.dtype == np.uint8, f"{name} LUT must be uint8, got {lut.dtype}"
        for required in ("viridis", "inferno", "turbo", "ndvi", "ndwi", "ndbi", "gray"):
            assert required in colormaps.COLORMAP_STOPS, f"{required} must be registered"

    def test_unknown_name_falls_back_to_viridis(self):
        assert np.array_equal(
            colormaps.get_colormap("not-a-colormap"), colormaps.get_colormap("viridis")
        ), "an unknown colormap name must fall back to viridis"

    def test_degenerate_inputs_do_not_raise(self):
        all_nan = np.full((8, 8), np.nan, dtype=np.float32)
        rgb = colormaps.apply_colormap(all_nan, "viridis", nodata_rgb=(7, 8, 9))
        assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
        assert np.all(rgb == np.array([7, 8, 9], dtype=np.uint8)), "all-NaN must be all nodata"

        constant = colormaps.apply_colormap(np.full((8, 8), 0.5, dtype=np.float32), "inferno")
        assert constant.shape == (8, 8, 3)
        assert len(np.unique(constant.reshape(-1, 3), axis=0)) == 1, "constant input, one colour"

        ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        pinned = colormaps.apply_colormap(ramp, "turbo", vmin=0.5, vmax=0.5)
        assert pinned.shape == (8, 8, 3), "vmin == vmax must degrade gracefully"

    def test_rgba_alpha_marks_invalid_pixels(self):
        arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = np.nan
        mask = np.ones((8, 8), dtype=bool)
        mask[1, 1] = False

        rgba = colormaps.apply_colormap_rgba(arr, "ndvi", mask=mask)
        assert rgba.shape == (8, 8, 4) and rgba.dtype == np.uint8
        assert rgba[0, 0, 3] == 0, "non-finite pixels must be fully transparent"
        assert rgba[1, 1, 3] == 0, "masked-out pixels must be fully transparent"
        assert rgba[4, 4, 3] == 255, "valid pixels must be fully opaque"

    def test_stops_hex(self):
        for name in colormaps.COLORMAP_STOPS:
            stops = colormaps.colormap_stops_hex(name, 9)
            assert len(stops) == 9, f"{name} must yield exactly 9 legend stops"
            for value in stops:
                assert HEX_RE.match(value), f"{name} produced malformed hex {value!r}"


class TestWaldProtocol:
    """Wald's synthesis protocol and the bicubic reference baseline."""

    def test_mtf_degrade(self, lr_stack: torch.Tensor):
        degraded = consistency.mtf_degrade(lr_stack, factor=4)
        assert degraded.shape == (10, 8, 8), "mtf_degrade must divide both axes by the factor"
        assert torch.isfinite(degraded).all(), "degradation must not introduce NaNs"
        assert float(degraded.min()) >= 0.0
        assert float(degraded.max()) <= 1.0
        assert float(degraded.mean()) == pytest.approx(float(lr_stack.mean()), abs=0.02)

    def test_bicubic_model_matches_its_own_baseline(self, wald_result):
        assert wald_result.protocol == "Wald synthesis (10 m -> 40 m -> 10 m)"
        assert wald_result.scale_factor == 4
        assert wald_result.reference_shape == [10, 32, 32]
        assert wald_result.degraded_shape == [10, 8, 8]
        assert wald_result.band_names == S2_10BAND_NAMES
        assert wald_result.metrics["psnr_db"] == pytest.approx(
            wald_result.baseline_metrics["psnr_db"], abs=1e-6
        ), "a bicubic model cannot beat the bicubic baseline"
        for key, gain in wald_result.improvement.items():
            assert gain == pytest.approx(0.0, abs=1e-6), f"{key} gain must be ~0 for bicubic"

    def test_bad_model_scores_worse_than_baseline(self, lr_stack: torch.Tensor):
        bad = consistency.wald_protocol_validate(lr_stack, nearest_predict, factor=4)
        assert bad.metrics["psnr_db"] < bad.baseline_metrics["psnr_db"], (
            "nearest-neighbour upsampling must lose to bicubic on PSNR"
        )
        for key in ("psnr_db", "ssim", "scc", "rmse", "sam_deg", "ergas"):
            assert bad.improvement[key] < 0.0, (
                f"improvement[{key!r}] must be negative for a worse-than-bicubic model"
            )
        assert bad.improvement["psnr_db"] < -0.1, "the gap must be measurable, not numerical noise"

    def test_to_dict_is_json_serialisable(self, wald_result):
        payload = wald_result.to_dict()
        assert set(payload) >= {"metrics", "baseline_metrics", "improvement", "protocol",
                                "reference_shape", "degraded_shape", "scale_factor"}
        json.dumps(payload, allow_nan=False)


class TestConsistency:
    """Wald's consistency property and per-band spectral bookkeeping."""

    def test_bicubic_product_round_trips(self, lr_stack: torch.Tensor, sr_stack: torch.Tensor):
        result = consistency.consistency_check(lr_stack, sr_stack, factor=4)
        assert result.passed, (
            f"a bicubic product must satisfy consistency; bias={result.max_abs_bias:.5f}, "
            f"sam={result.spectral_angle_deg:.3f} deg"
        )
        assert result.max_abs_bias < result.tolerance["max_abs_bias"]
        assert len(result.per_band_bias) == 10
        assert len(result.per_band_rmse) == 10
        assert result.band_names == S2_10BAND_NAMES
        json.dumps(result.to_dict(), allow_nan=False)

    def test_reflectance_offset_is_caught_with_the_right_sign(
        self, lr_stack: torch.Tensor, sr_stack: torch.Tensor
    ):
        biased = consistency.consistency_check(lr_stack, sr_stack + 0.05, factor=4)
        assert not biased.passed, "a +0.05 reflectance offset must fail the consistency check"
        assert biased.max_abs_bias == pytest.approx(0.05, abs=5e-3)
        for name, bias in zip(biased.band_names, biased.per_band_bias):
            assert bias > 0.0, f"{name} bias must be positive for a brightened product"
            assert bias == pytest.approx(0.05, abs=5e-3)

        darkened = consistency.consistency_check(lr_stack, sr_stack - 0.05, factor=4)
        assert all(bias < 0.0 for bias in darkened.per_band_bias), (
            "a darkened product must report a negative bias"
        )

    def test_spectral_fidelity_table(self, lr_stack: torch.Tensor, sr_stack: torch.Tensor):
        table = consistency.spectral_fidelity(lr_stack, sr_stack, factor=4)
        assert set(table) == {"bands", "max_rel_pct", "mean_abs_delta"}
        assert len(table["bands"]) == 10, "one row per band"
        assert [row["band"] for row in table["bands"]] == S2_10BAND_NAMES
        for row in table["bands"]:
            assert set(row) == {"band", "lr_mean", "sr_mean", "delta", "rel_pct"}
            assert row["delta"] == pytest.approx(row["sr_mean"] - row["lr_mean"], abs=1e-9)
        assert table["mean_abs_delta"] < 0.01, "bicubic must preserve the mean reflectance"


class TestUncertainty:
    """Test-time-augmentation spread, novelty and the hallucination-risk summary."""

    def test_d4_transforms_round_trip(self):
        pairs = uncertainty.d4_transforms()
        assert len(pairs) == 8, "the dihedral group D4 has exactly 8 elements"
        sample = torch.rand(3, 8, 10, generator=torch.Generator().manual_seed(11))
        for index, (forward, inverse) in enumerate(pairs):
            restored = inverse(forward(sample))
            assert torch.allclose(restored, sample), f"D4 element {index} is not invertible"
        assert torch.allclose(pairs[0][0](sample), sample), "element 0 must be the identity"

    def test_tta_ensemble_shape(self, lr_stack: torch.Tensor):
        members = uncertainty.tta_ensemble(lr_stack, bicubic_predict, n_members=4)
        assert members.shape == (4, 10, 128, 128)
        assert torch.isfinite(members).all()
        assert uncertainty.tta_ensemble(lr_stack, bicubic_predict, n_members=99).shape[0] == 8, (
            "n_members must be clamped to at most 8"
        )

    def test_ensemble_and_novelty_only_paths(self, lr_stack: torch.Tensor, sr_stack: torch.Tensor):
        ensemble = uncertainty.estimate_uncertainty(
            lr_stack, sr_stack, predict_fn=bicubic_predict, n_ensemble=4
        )
        assert ensemble.method == "tta-d4-ensemble+novelty"
        assert ensemble.n_ensemble == 4
        assert ensemble.band_std is not None and ensemble.band_std.shape == (10, 128, 128)
        assert ensemble.mean_std < 1e-3, (
            "a rotation-equivariant predictor must show almost no ensemble spread"
        )

        free = uncertainty.estimate_uncertainty(lr_stack, sr_stack)
        assert free.method == "novelty-only"
        assert free.n_ensemble == 1
        assert free.band_std is None
        assert not free.std_map.any(), "the novelty-only path must report a zero spread map"

    def test_confidence_and_risk_bounds(self, lr_stack: torch.Tensor):
        rng = np.random.default_rng(3)
        invented = consistency.upsample_bicubic(lr_stack, factor=4).numpy()
        invented = invented + 0.05 * rng.normal(0.0, 1.0, invented.shape).astype(np.float32)
        result = uncertainty.estimate_uncertainty(lr_stack, invented)

        assert result.confidence_map.shape == (128, 128)
        assert float(result.confidence_map.min()) >= 0.0
        assert float(result.confidence_map.max()) <= 1.0
        assert 0.0 <= result.reliability_score <= 100.0
        assert result.hallucination_risk in {"low", "moderate", "elevated", "high"}
        assert result.mean_novelty > 0.0, "synthesised detail must register as novelty"
        assert result.interpretation.strip(), "the UI needs a plain-English interpretation"

    def test_to_dict_has_no_arrays(self, lr_stack: torch.Tensor, sr_stack: torch.Tensor):
        payload = uncertainty.estimate_uncertainty(lr_stack, sr_stack).to_dict()
        assert not any(isinstance(value, np.ndarray) for value in payload.values()), (
            "to_dict must stay lightweight: no full-resolution arrays"
        )
        assert set(payload) >= {"method", "n_ensemble", "reliability_score",
                                "hallucination_risk", "histogram"}
        assert len(payload["histogram"]["counts"]) == 32
        assert len(payload["histogram"]["edges"]) == 33
        json.dumps(payload, allow_nan=False)

    def test_render_png(self, lr_stack: torch.Tensor, sr_stack: torch.Tensor, tmp_path: Path):
        result = uncertainty.estimate_uncertainty(lr_stack, sr_stack)
        for kind in ("confidence", "std", "novelty"):
            path = uncertainty.render_uncertainty_png(result, tmp_path / f"{kind}.png", kind=kind)
            assert Path(path).is_file(), f"the {kind} PNG was not written"
            assert Path(path).stat().st_size > 0, f"the {kind} PNG is empty"
        with pytest.raises(ValueError):
            uncertainty.render_uncertainty_png(result, tmp_path / "bad.png", kind="nope")


class TestSpectralIndices:
    """Application-layer spectral index products."""

    def test_vegetation_spectrum(self):
        stack = _spectrum_stack(VEGETATION_SPECTRUM)
        ndvi = indices.compute_index(stack, "ndvi")
        assert ndvi.shape == (16, 16) and ndvi.dtype == np.float32
        assert 0.7 < float(np.nanmean(ndvi)) < 0.9, "dense vegetation must give a high NDVI"
        assert float(np.nanmean(indices.compute_index(stack, "ndbi"))) < 0.0, (
            "vegetation must not be flagged as built-up"
        )
        assert float(np.nanmean(indices.compute_index(stack, "ndre"))) > 0.3

    def test_water_spectrum(self):
        stack = _spectrum_stack(WATER_SPECTRUM)
        assert float(np.nanmean(indices.compute_index(stack, "ndwi"))) > 0.5, (
            "open water must give a strongly positive NDWI"
        )
        assert float(np.nanmean(indices.compute_index(stack, "mndwi"))) > 0.5
        assert float(np.nanmean(indices.compute_index(stack, "ndvi"))) < 0.0

    def test_every_registered_index(self, sr_stack: torch.Tensor):
        expected = {"ndvi", "ndre", "savi", "evi", "ndwi", "mndwi", "ndbi", "bsi",
                    "nbr", "ndmi"}
        assert expected <= set(indices.INDEX_REGISTRY), "the registry misses required indices"

        computed = indices.compute_indices(sr_stack)
        assert set(computed) == set(indices.INDEX_REGISTRY)
        for key, arr in computed.items():
            assert arr.shape == (128, 128), f"{key} must keep the SR grid"
            stats = indices.index_statistics(arr, key)
            assert set(stats) >= {"key", "name", "mean", "std", "min", "max", "p05",
                                  "p50", "p95", "valid_fraction", "classes"}, (
                f"the {key} statistics are incomplete")
            total = sum(entry["fraction"] for entry in stats["classes"])
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"{key} class fractions sum to {total}, expected 1.0"
            )
            for entry in stats["classes"]:
                assert HEX_RE.match(entry["color"]), f"the {key} class colour is malformed"

    def test_error_handling(self, sr_stack: torch.Tensor):
        with pytest.raises(KeyError):
            indices.compute_index(sr_stack, "not-an-index")
        rgbn = sr_stack[[0, 1, 2, 6]]
        with pytest.raises(ValueError):
            indices.compute_index(rgbn, "ndbi", band_names=["B02", "B03", "B04", "B08"])

    def test_render_and_registry_json(self, sr_stack: torch.Tensor, tmp_path: Path):
        ndvi = indices.compute_index(sr_stack, "ndvi")
        path = indices.render_index_png(ndvi, "ndvi", tmp_path / "ndvi.png")
        assert Path(path).is_file() and Path(path).stat().st_size > 0, "no NDVI PNG was written"

        delta = indices.index_delta_statistics(
            indices.compute_index(_synthetic_stack(), "ndvi"), ndvi, "ndvi"
        )
        assert set(delta) >= {"mean_lr", "mean_sr", "mean_abs_delta", "edge_gain"}

        payload = indices.registry_as_json()
        assert len(payload) == len(indices.INDEX_REGISTRY)
        for entry in payload:
            assert entry["legend_hex"], f"{entry['key']} has no legend gradient"
            assert all(HEX_RE.match(value) for value in entry["legend_hex"])
        json.dumps(payload, allow_nan=False)


class TestQualityReport:
    """Assembly of the analyst-facing QA sheet."""

    @staticmethod
    def _build(passed: bool, psnr_db: float, ssim: float, reliability: float):
        """Build a QualityReport from hand-written module payloads."""
        wald = {
            "metrics": {"psnr_db": psnr_db, "ssim": ssim, "rmse": 0.02, "sam_deg": 1.2},
            "baseline_metrics": {"psnr_db": psnr_db - 2.0, "ssim": ssim - 0.05,
                                 "rmse": 0.03, "sam_deg": 1.6},
            "improvement": {"psnr_db": 2.0, "ssim": 0.05, "rmse": 0.01, "sam_deg": 0.4},
            "reference_shape": [10, 32, 32], "degraded_shape": [10, 8, 8],
            "scale_factor": 4, "protocol": "Wald synthesis (10 m -> 40 m -> 10 m)",
            "band_names": list(S2_10BAND_NAMES),
        }
        cons = {
            "metrics": {"psnr_db": 45.0, "ssim": 0.99},
            "per_band_bias": [0.001] * 10, "per_band_rmse": [0.002] * 10,
            "max_abs_bias": 0.001 if passed else 0.05,
            "spectral_angle_deg": 0.4 if passed else 4.0,
            "passed": passed, "band_names": list(S2_10BAND_NAMES),
            "tolerance": {"max_abs_bias": 0.01, "spectral_angle_deg": 1.0},
        }
        unc = {
            "method": "tta-d4-ensemble+novelty", "n_ensemble": 4,
            "reliability_score": reliability, "mean_std": 0.002, "mean_novelty": 0.01,
            "hallucination_risk": "low" if reliability >= 85.0 else "high",
        }
        return report.build_report(
            job_id="job-test",
            scene={"tile": "T43RGQ", "acquired": "2024-03-01"},
            wald=wald,
            consistency=cons,
            uncertainty=unc,
        )

    def test_verdicts(self):
        assert self._build(True, 38.0, 0.95, 92.0).summary["verdict"] == "validated"
        assert self._build(True, 27.0, 0.78, 68.0).summary["verdict"] == "validated-with-caveats"
        assert self._build(False, 18.0, 0.45, 40.0).summary["verdict"] == "inconclusive"

    def test_mandatory_caveat_always_present(self):
        for quality in (
            self._build(True, 38.0, 0.95, 92.0),
            self._build(False, 18.0, 0.45, 40.0),
        ):
            caveats = " ".join(quality.summary["caveats"]).lower()
            assert "not directly observed" in caveats, (
                "the scientific-reconstruction notice must always be present"
            )

    def test_to_dict_and_markdown(self):
        quality = self._build(True, 38.0, 0.95, 92.0)
        payload = quality.to_dict()
        assert set(payload) >= {"job_id", "generated_at", "scene", "wald", "consistency",
                                "uncertainty", "summary"}
        json.dumps(payload, allow_nan=False)

        markdown = quality.to_markdown()
        assert markdown.lstrip().startswith("#"), "the QA sheet must open with a heading"
        for heading in ("Wald", "Consistency", "Uncertainty"):
            assert heading.lower() in markdown.lower(), f"the {heading} section is missing"
        assert "|" in markdown and "---" in markdown, "the QA sheet must contain markdown tables"
        assert "job-test" in markdown

    def test_save_report(self, tmp_path: Path):
        quality = self._build(True, 38.0, 0.95, 92.0)
        paths = report.save_report(quality, tmp_path)
        assert set(paths) == {"json", "markdown"}

        json_path = Path(paths["json"])
        md_path = Path(paths["markdown"])
        assert json_path.name == "quality_report.json"
        assert md_path.name == "quality_report.md"
        assert json.loads(json_path.read_text(encoding="utf-8"))["job_id"] == "job-test"
        assert md_path.read_text(encoding="utf-8").strip(), "the markdown report is empty"
