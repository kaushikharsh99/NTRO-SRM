"""Tests for SIH Round-2 inference optimizations.

Covers: vectorized Fourier filters vs loop reference, mask cache reuse,
tile-grid coverage (square/rectangular/small), blending-window properties,
batched tiled equivalence with a mock model, and rectangular model inference.
"""

import torch

from ntro_srm.inference.tiled import (
    TiledInferenceConfig,
    blend_window,
    compute_tile_grid,
    tiled_predict,
)
from ntro_srm.models import fourier_filters as FF


class TestVectorizedFourierFilters:
    def _loop_ideal(self, h, w, cutoff):
        crow, ccol = h // 2, w // 2
        m = torch.zeros((h, w), dtype=torch.float32)
        for u in range(h):
            for v in range(w):
                if ((u - crow) ** 2 + (v - ccol) ** 2) ** 0.5 <= cutoff:
                    m[u, v] = 1
        return m

    def test_ideal_matches_loop_rectangular(self):
        for shape in [(16, 16), (16, 24), (24, 16), (32, 20)]:
            ref = self._loop_ideal(*shape, 5)
            vec = FF.ideal_filter(shape, 5)
            assert torch.equal(ref, vec), f"mismatch for {shape}"

    def test_all_filters_finite_rectangular(self):
        shape = (20, 32)
        assert torch.isfinite(FF.butterworth_filter(shape, 6, order=2)).all()
        assert torch.isfinite(FF.gaussian_filter(shape, 6)).all()
        assert torch.isfinite(FF.sigmoid_filter(shape, 6, sharpness=8.0)).all()
        g = FF.gaussian_filter(shape, 6)
        assert float(g.max()) <= 1.0 + 1e-6 and float(g.min()) >= 0.0

    def test_mask_cache_reuses_storage(self):
        FF.clear_mask_cache()
        a = FF.get_low_pass_mask("ideal", (32, 48), cutoff=8, device="cpu")
        b = FF.get_low_pass_mask("ideal", (32, 48), cutoff=8, device="cpu")
        assert a is b
        assert FF.cache_stats()["entries"] >= 1

    def test_hard_constraint_matches_reference(self):
        torch.manual_seed(0)
        lr = torch.rand(1, 4, 32, 32) * 0.5
        sr = torch.rand(1, 4, 64, 64) * 0.5
        low = FF.get_low_pass_mask("gaussian", (64, 64), cutoff=16, device="cpu")
        out = FF.apply_hard_constraint(lr, sr, low)
        assert out.shape == sr.shape
        assert torch.isfinite(out).all()
        # Reference computation with explicit high-pass.
        lr_up = torch.nn.functional.interpolate(lr, size=(64, 64), mode="bicubic", antialias=True)
        ref = torch.real(
            torch.fft.ifft2(
                torch.fft.ifftshift(
                    torch.fft.fftshift(torch.fft.fft2(lr_up.float()), dim=(-2, -1)) * low
                    + torch.fft.fftshift(torch.fft.fft2(sr.float()), dim=(-2, -1)) * (1 - low),
                    dim=(-2, -1),
                )
            )
        )
        assert torch.allclose(out.float(), ref, atol=1e-5)


class TestTileGrid:
    def _covers(self, h, w, tile, overlap):
        grid = compute_tile_grid(h, w, tile, overlap)
        assert len(grid) == len(set(grid)), "duplicate tiles emitted"
        covered = torch.zeros(h, w, dtype=torch.bool)
        for y, x in grid:
            assert 0 <= y <= h - tile or (h <= tile and y == 0)
            assert 0 <= x <= w - tile or (w <= tile and x == 0)
            covered[y : y + min(tile, h), x : x + min(tile, w)] = True
        assert covered.all(), f"incomplete coverage for {(h, w)}"
        return grid

    def test_small_single_tile(self):
        assert compute_tile_grid(64, 64, 128, 32) == [(0, 0)]
        assert compute_tile_grid(128, 128, 128, 32) == [(0, 0)]

    def test_square_coverage(self):
        grid = self._covers(256, 256, 128, 32)
        # stride 96 -> per-axis origins [0, 96, 128], i.e. 3x3 = 9 unique tiles.
        assert len(grid) == 9
        assert (128, 128) in grid

    def test_rectangular_coverage(self):
        for h, w in [(130, 140), (192, 320), (300, 160), (100, 300)]:
            grid = self._covers(h, w, 128, 32)
            assert len(grid) >= 2

    def test_non_divisible_no_duplicates(self):
        grid = self._covers(200, 200, 128, 32)
        # stride 96: naive range gives 0,96,192 -> 192 clamped to 72; must dedup.
        assert (72, 72) in grid
        assert len(grid) == 4


class TestBlendWindow:
    def test_borders_stay_one(self):
        win = blend_window(64, 16, False, False, False, False, "linear")
        assert torch.allclose(win, torch.ones_like(win))

    def test_interior_partitions_unity(self):
        # Adjacent tiles: right ramp of left tile + left ramp of right tile == 1.
        ov = 16
        left = blend_window(64, ov, False, False, False, True, "linear")
        right = blend_window(64, ov, False, True, False, False, "linear")
        assert torch.allclose(left[:, -ov:] + right[:, :ov], torch.ones(64, ov), atol=1e-5)

    def test_hann_smooth(self):
        win = blend_window(64, 16, True, True, True, True, "hann")
        assert bool((win >= 0).all() and (win <= 1.0 + 1e-6).all())
        assert float(win.max()) > 0.99


class TestTiledPredict:
    def _bicubic_up(self, scale=4):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(
                x, scale_factor=scale, mode="bicubic", align_corners=False, antialias=True
            )

        return fn

    def test_constant_input_exact(self):
        lr = torch.full((1, 3, 200, 150), 0.37)
        out = tiled_predict(
            self._bicubic_up(), lr,
            TiledInferenceConfig(tile_size=64, overlap=16, batch_size=4, blend_mode="linear"),
        )
        assert out.shape == (1, 3, 800, 600)
        assert torch.allclose(out, torch.full_like(out, 0.37), atol=2e-3)

    def test_matches_single_forward_when_one_tile(self):
        torch.manual_seed(1)
        lr = torch.rand(2, 3, 64, 64)
        cfg = TiledInferenceConfig(tile_size=128, overlap=32, batch_size=2)
        out = tiled_predict(self._bicubic_up(), lr, cfg)
        ref = self._bicubic_up()(torch.nn.functional.pad(lr, (0, 64, 0, 64), mode="replicate"))[:, :, :256, :256]
        assert torch.allclose(out, ref, atol=1e-5)

    def test_batch_equivalence(self):
        torch.manual_seed(2)
        lr = torch.rand(1, 3, 160, 160)
        cfg1 = TiledInferenceConfig(tile_size=64, overlap=16, batch_size=1, blend_mode="linear")
        cfg4 = TiledInferenceConfig(tile_size=64, overlap=16, batch_size=8, blend_mode="linear")
        assert torch.allclose(tiled_predict(self._bicubic_up(), lr, cfg1),
                              tiled_predict(self._bicubic_up(), lr, cfg4), atol=1e-6)

    def test_rectangular_output_shape(self):
        lr = torch.rand(1, 10, 130, 140)
        out = tiled_predict(
            self._bicubic_up(), lr,
            TiledInferenceConfig(tile_size=128, overlap=32, batch_size=2),
        )
        assert out.shape == (1, 10, 520, 560)
