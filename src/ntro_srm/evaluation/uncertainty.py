"""Uncertainty and hallucination-risk estimation for super-resolved Sentinel-2 scenes.

A 4x super-resolution network does not *observe* the 2.5 m detail it produces: it
infers that detail from a 10 m observation plus everything it learned during
training. Some of the output is therefore evidence and some of it is an educated
guess, and an analyst has no way of telling the two apart by eye.

This module makes that distinction explicit and spatial. It combines two
independent signals:

* **Ensemble spread** -- the same scene is super-resolved several times under the
  eight flip/rotate symmetries of the square (the dihedral group D4). A genuinely
  observed edge is reconstructed identically no matter how the tile is oriented;
  invented texture is not, so the per-pixel standard deviation across the ensemble
  measures how much of the result is an artefact of the model's viewpoint.
* **Novelty** -- the magnitude of the detail the network added on top of a plain
  bicubic interpolation of the observation. Bicubic contains no information the
  sensor did not record, so anything above it is, by construction, synthesised.

The two are fused into a per-pixel ``confidence_map`` in [0, 1] (1 = trustworthy,
0 = largely invented), a scene-level ``reliability_score`` out of 100, and a plain
language ``interpretation`` written for the analyst who has to sign off on the
product.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# The shared (C, H, W) coercion helper lives in the sibling ``_common`` module. It
# is imported softly so this module stays usable, and testable, on its own.
try:  # pragma: no cover - trivial import guard
    from ntro_srm.evaluation._common import to_chw_numpy as _to_chw_numpy
except ImportError:  # pragma: no cover - trivial import guard
    _to_chw_numpy = None

ArrayLike = Union[np.ndarray, torch.Tensor]
PredictFn = Callable[[torch.Tensor], torch.Tensor]

# --------------------------------------------------------------------------- #
# Fusion weights and reporting thresholds (documented module constants)
# --------------------------------------------------------------------------- #

# Weight of the normalized ensemble spread in the confidence map. Ensemble
# disagreement is the stronger evidence of invention, being a direct probe of the
# model itself, so it carries the larger share whenever an ensemble was run.
W_STD: float = 0.6

# Weight of the normalized novelty map when an ensemble is available.
W_NOVELTY: float = 0.4

# Fallback weights used when no `predict_fn` was supplied and the novelty map is
# the only evidence available (method == "novelty-only").
W_STD_NOVELTY_ONLY: float = 0.0
W_NOVELTY_ONLY: float = 1.0

# Both terms are normalized by their own 99th percentile before fusion, so the
# weights above are comparable and the confidence map stays scene-relative.
NORMALIZATION_PERCENTILE: float = 99.0

# Reflectance magnitude below which a map is treated as identically zero. One
# Sentinel-2 L2A quantisation step is 1/10000 reflectance, so a spread or a novelty
# signal smaller than this is below what the sensor can even represent -- it is
# float round-off, not evidence. Without this floor the scene-relative
# normalization above would rescale numerical noise to fill [0, 1] and report a
# pixel-perfect reconstruction as untrustworthy.
NEGLIGIBLE_MAGNITUDE: float = 1e-4

# A pixel below this confidence is quoted to the analyst as "low confidence".
LOW_CONFIDENCE_THRESHOLD: float = 0.5

# reliability_score (0-100) -> hallucination-risk band, evaluated top-down.
HALLUCINATION_RISK_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (85.0, "low"),
    (70.0, "moderate"),
    (55.0, "elevated"),
)
HALLUCINATION_RISK_FALLBACK: str = "high"


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #


def _as_chw_tensor(array: ArrayLike, name: str = "tensor") -> torch.Tensor:
    """Coerce an array or tensor to a contiguous (C, H, W) float32 CPU tensor."""
    if isinstance(array, torch.Tensor):
        tensor = array.detach().to(device="cpu", dtype=torch.float32)
    else:
        data = np.asarray(array)
        if _to_chw_numpy is not None:
            data = _to_chw_numpy(data, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Batched input is not supported for {name}: expected a leading "
                f"dimension of 1, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor[0]
    elif tensor.ndim != 3:
        raise ValueError(
            f"Expected (H, W), (C, H, W) or (1, C, H, W) for {name}, "
            f"got shape {tuple(tensor.shape)}"
        )

    return tensor.contiguous()


def _finite(values: np.ndarray) -> np.ndarray:
    """Flatten to the finite values only (may be empty)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


def _safe_mean(values: np.ndarray, default: float = float("nan")) -> float:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else default


def _safe_max(values: np.ndarray, default: float = float("nan")) -> float:
    finite = _finite(values)
    return float(finite.max()) if finite.size else default


def _safe_percentile(values: np.ndarray, q: float, default: float = float("nan")) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, q)) if finite.size else default


def _box_blur(image: torch.Tensor, size: int) -> torch.Tensor:
    """Replicate-padded `size` x `size` box filter on a (H, W) tensor."""
    if size <= 1:
        return image
    left = (size - 1) // 2
    right = size // 2
    padded = F.pad(image[None, None], (left, right, left, right), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=size, stride=1)[0, 0]


def _normalize_by_percentile(array: np.ndarray, q: float = NORMALIZATION_PERCENTILE) -> np.ndarray:
    """Scale by the array's own `q`-th percentile and clip to [0, 1].

    A degenerate scale -- non-finite, or no larger than ``NEGLIGIBLE_MAGNITUDE``
    (a perfectly flat map, an entirely invalid one, or one holding nothing but
    float round-off) -- yields all zeros, i.e. "no evidence of invention".
    """
    out = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = _safe_percentile(out, q, default=0.0)
    if not np.isfinite(scale) or scale <= NEGLIGIBLE_MAGNITUDE:
        return np.zeros_like(out, dtype=np.float32)
    return np.clip(out / float(scale), 0.0, 1.0).astype(np.float32)


def _risk_band(reliability_score: float) -> str:
    """Map a 0-100 reliability score onto its hallucination-risk band."""
    if not np.isfinite(reliability_score):
        return HALLUCINATION_RISK_FALLBACK
    for threshold, label in HALLUCINATION_RISK_THRESHOLDS:
        if reliability_score >= threshold:
            return label
    return HALLUCINATION_RISK_FALLBACK


def _build_interpretation(
    reliability_score: float,
    risk: str,
    method: str,
    n_ensemble: int,
    low_confidence_pct: float,
    valid_fraction: float = 1.0,
) -> str:
    """Compose the analyst-facing paragraph shipped with every product."""
    score_txt = f"{reliability_score:.1f}" if np.isfinite(reliability_score) else "not available"
    low_txt = f"{low_confidence_pct:.0f}%" if np.isfinite(low_confidence_pct) else "an unknown share"

    if not np.isfinite(valid_fraction) or valid_fraction <= 0.0:
        # No observation to reason about: say so instead of reporting the
        # vacuously perfect score that an empty scene would otherwise produce.
        return (
            "No valid pixels were found in this scene: every position is masked or "
            "non-finite across all bands, so there is nothing to compare against the "
            f"observation and the reliability score of {score_txt} out of 100 is "
            "vacuous rather than reassuring. Treat the whole product as unverified, "
            "check the cloud mask, the nodata handling and the tile footprint, and "
            "re-run the assessment on a scene that actually carries data."
        )

    if method == "novelty-only":
        opening = (
            f"Reliability score {score_txt} out of 100, placing this scene in the "
            f"'{risk}' hallucination-risk band; it was derived from the novelty map "
            f"alone because no ensemble was run, so it measures how much detail the "
            f"network added beyond the observation but not how stable that detail is, "
            f"and it should be read as an optimistic bound on the true uncertainty."
        )
        middle = (
            f"About {low_txt} of the super-resolved pixels fall below a confidence of "
            f"{LOW_CONFIDENCE_THRESHOLD:.1f}, concentrated wherever the model sharpened "
            f"edges and textures that the 10 m bands recorded only as a blur."
        )
    else:
        opening = (
            f"Reliability score {score_txt} out of 100, placing this scene in the "
            f"'{risk}' hallucination-risk band, from a {n_ensemble}-member D4 "
            f"test-time-augmentation ensemble fused with the novelty map."
        )
        middle = (
            f"About {low_txt} of the super-resolved pixels fall below a confidence of "
            f"{LOW_CONFIDENCE_THRESHOLD:.1f}: there the ensemble members disagreed with "
            f"one another, and/or the network added structure that is simply not present "
            f"in the interpolated 10 m observation."
        )

    closing = (
        "Bright, low-confidence areas on the uncertainty overlays are reconstructed "
        "rather than observed, being the model's most plausible guess at detail that "
        "Sentinel-2 never resolved, so they must not be used on their own as evidence "
        "for detection, identification, counting or measurement, and any finding that "
        "rests on them needs corroboration from an independent higher-resolution or "
        "ground-truth source before it is reported."
    )

    if valid_fraction < 0.999:
        closing += (
            f" Note also that only {100.0 * valid_fraction:.0f}% of the scene carries "
            "valid data; the masked remainder is excluded from these figures and is "
            "not covered by this assessment."
        )
    return f"{opening} {middle} {closing}"


# --------------------------------------------------------------------------- #
# D4 test-time augmentation
# --------------------------------------------------------------------------- #


def _make_d4_pair(k: int, mirror: bool) -> tuple[Callable, Callable]:
    """Build one (forward, inverse) D4 pair: optional mirror, then `k` rot90 turns."""

    def forward(tensor: torch.Tensor, _k: int = k, _m: bool = mirror) -> torch.Tensor:
        out = torch.flip(tensor, dims=[-1]) if _m else tensor
        return torch.rot90(out, _k, dims=(-2, -1)) if _k else out

    def inverse(tensor: torch.Tensor, _k: int = k, _m: bool = mirror) -> torch.Tensor:
        out = torch.rot90(tensor, -_k, dims=(-2, -1)) if _k else tensor
        return torch.flip(out, dims=[-1]) if _m else out

    return forward, inverse


def d4_transforms() -> list[tuple[Callable, Callable]]:
    """Return the eight symmetries of the square as (forward, inverse) pairs.

    The dihedral group D4 -- four rotations, plus those same four rotations composed
    with a horizontal mirror -- is the complete set of orientations under which a
    faithful reconstruction must be invariant. Each pair acts on the last two
    dimensions of a (C, H, W) tensor, so a batched (B, C, H, W) tensor works too.
    Both members are pure index permutations (``torch.flip`` / ``torch.rot90``) with
    no interpolation and no arithmetic, hence ``inverse(forward(x))`` restores `x`
    bit-for-bit.

    Returns
    -------
    list of (Callable, Callable)
        Exactly 8 ``(forward, inverse)`` pairs. Element 0 is the identity, elements
        1-3 are the 90/180/270 degree rotations, and elements 4-7 are the mirrored
        counterparts of those four. Truncating the list from the front therefore
        always yields a valid, orientation-diverse subset.
    """
    pairs: list[tuple[Callable, Callable]] = []
    for mirror in (False, True):
        for k in range(4):
            pairs.append(_make_d4_pair(k, mirror))
    return pairs


def tta_ensemble(
    lr_tensor: ArrayLike,
    predict_fn: PredictFn,
    n_members: int = 4,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> torch.Tensor:
    """Super-resolve one scene under several D4 orientations and undo each transform.

    Each member re-orients the low-resolution input, runs the network on that
    re-oriented view, and maps the prediction back to the original orientation. All
    members are then directly comparable pixel by pixel: where they agree, the
    reconstruction is driven by the observation; where they disagree, it is driven
    by the model.

    Parameters
    ----------
    lr_tensor : np.ndarray or torch.Tensor
        Low-resolution scene of shape (C, H, W) with reflectance in [0, 1].
    predict_fn : Callable
        Super-resolution callable mapping a (C, H, W) tensor to (C, factor*H, factor*W).
    n_members : int, default=4
        Number of D4 orientations to run, clamped to [1, 8]. 1 reproduces a plain
        single forward pass; 8 uses the full group.
    progress_callback : Callable, optional
        Called as ``progress_callback(message, percent)`` after each member
        completes, with ``message = "TTA member i/n"`` and ``percent`` the share of
        members finished, from 0.0 to 100.0.

    Returns
    -------
    torch.Tensor
        Stacked predictions of shape (n_members, C, factor*H, factor*W), float32,
        all returned to the original orientation.
    """
    lr = _as_chw_tensor(lr_tensor, name="lr_tensor")
    n_members = int(max(1, min(8, int(n_members))))
    pairs = d4_transforms()[:n_members]

    members: list[torch.Tensor] = []
    with torch.no_grad():
        for index, (forward, inverse) in enumerate(pairs):
            prediction = predict_fn(forward(lr))
            if not isinstance(prediction, torch.Tensor):
                prediction = torch.as_tensor(np.asarray(prediction), dtype=torch.float32)
            restored = inverse(prediction.detach().to(device="cpu", dtype=torch.float32))
            members.append(restored.contiguous())
            if progress_callback is not None:
                progress_callback(
                    f"TTA member {index + 1}/{n_members}",
                    100.0 * (index + 1) / n_members,
                )

    return torch.stack(members, dim=0)


# --------------------------------------------------------------------------- #
# Novelty (detail added beyond the observation)
# --------------------------------------------------------------------------- #


def novelty_map(
    lr_tensor: ArrayLike,
    sr_tensor: ArrayLike,
    factor: int = 4,
    smooth: int = 3,
) -> np.ndarray:
    """Map the detail the network invented on top of the observed signal.

    A bicubic interpolation of the 10 m scene carries no information the sensor did
    not record. Everything the super-resolved product contains *above* that
    interpolation was synthesised by the network, so the band-averaged absolute
    difference between the two is a direct, free measure of how much of each pixel
    is model-derived rather than observed. A small box filter suppresses
    single-pixel noise so the result reads as coherent regions on screen.

    Parameters
    ----------
    lr_tensor : np.ndarray or torch.Tensor
        Observed low-resolution scene, (C, H, W), reflectance in [0, 1].
    sr_tensor : np.ndarray or torch.Tensor
        Super-resolved scene, (C, factor*H, factor*W), same band order.
    factor : int, default=4
        Nominal upscaling factor, kept for API symmetry with the rest of the
        evaluation package. The resampling always targets the exact grid of
        `sr_tensor`, so a mismatch degrades gracefully instead of raising.
    smooth : int, default=3
        Side length in pixels of the box filter. Values <= 1 disable smoothing.

    Returns
    -------
    np.ndarray
        Float32 array of shape (H_sr, W_sr) in reflectance units. Non-finite input
        pixels contribute 0.0 (treated as "no added detail") rather than poisoning
        their neighbourhood, so the map is always finite.

    Raises
    ------
    ValueError
        If `lr_tensor` and `sr_tensor` do not have the same number of bands.
    """
    lr = _as_chw_tensor(lr_tensor, name="lr_tensor")
    sr = _as_chw_tensor(sr_tensor, name="sr_tensor")

    if lr.shape[0] != sr.shape[0]:
        raise ValueError(
            f"Band count mismatch: lr_tensor has {lr.shape[0]} bands, "
            f"sr_tensor has {sr.shape[0]}"
        )

    target_h, target_w = int(sr.shape[-2]), int(sr.shape[-1])
    upsampled = F.interpolate(
        lr.unsqueeze(0),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )[0]

    difference = torch.nan_to_num(sr - upsampled, nan=0.0, posinf=0.0, neginf=0.0).abs()
    added_detail = difference.mean(dim=0)
    smoothed = _box_blur(added_detail, int(smooth))
    return smoothed.numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class UncertaintyResult:
    """Per-pixel and scene-level account of what is observed and what is inferred.

    Attributes
    ----------
    std_map : np.ndarray
        (H, W) ensemble standard deviation averaged over bands, in reflectance
        units. All zeros when no ensemble was run.
    band_std : np.ndarray or None
        (C, H, W) per-band ensemble standard deviation, or None when no ensemble was
        run. Useful for spotting bands (typically the resampled 20 m ones) that are
        reconstructed less reliably than the rest.
    novelty_map : np.ndarray
        (H, W) magnitude of the detail added beyond bicubic interpolation.
    confidence_map : np.ndarray
        (H, W) in [0, 1]. 1 means the pixel is essentially the observation carried
        through; 0 means it is largely invented by the network.
    mean_std, p95_std, max_std : float
        Scene statistics of `std_map`, in reflectance units.
    mean_novelty, p95_novelty : float
        Scene statistics of `novelty_map`, in reflectance units.
    reliability_score : float
        100 * mean(confidence_map); the single headline number, 0-100.
    hallucination_risk : str
        "low", "moderate", "elevated" or "high", banded from `reliability_score`.
    n_ensemble : int
        Number of ensemble members actually run (1 when none was run).
    method : str
        "tta-d4-ensemble+novelty" or "novelty-only".
    interpretation : str
        Plain-English paragraph for the quality report and the web UI.
    """

    std_map: np.ndarray
    band_std: np.ndarray | None
    novelty_map: np.ndarray
    confidence_map: np.ndarray
    mean_std: float
    p95_std: float
    max_std: float
    mean_novelty: float
    p95_novelty: float
    reliability_score: float
    hallucination_risk: str
    n_ensemble: int
    method: str
    interpretation: str

    def histogram(self, bins: int = 32) -> dict:
        """Distribution of per-pixel confidence, for the UI histogram widget.

        Parameters
        ----------
        bins : int, default=32
            Number of equal-width bins spanning the full [0, 1] confidence range, so
            that histograms from different scenes are directly comparable.

        Returns
        -------
        dict
            ``{"edges": [float, ...], "counts": [int, ...]}`` with
            ``len(edges) == bins + 1``. Non-finite pixels are excluded.
        """
        bins = int(max(1, int(bins)))
        finite = _finite(self.confidence_map)
        counts, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
        return {
            "edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
        }

    def to_dict(self) -> dict:
        """JSON-serialisable summary, without the full-resolution arrays.

        The maps themselves are delivered as PNG overlays by
        :func:`render_uncertainty_png`; this dictionary is what goes into the quality
        report and the API response.

        Returns
        -------
        dict
            Scene scalars, the method and its fusion weights, the risk band, the
            analyst-facing interpretation, and the confidence histogram.
        """
        height, width = int(self.confidence_map.shape[0]), int(self.confidence_map.shape[1])
        weights = (
            (W_STD_NOVELTY_ONLY, W_NOVELTY_ONLY)
            if self.method == "novelty-only"
            else (W_STD, W_NOVELTY)
        )
        return {
            "method": str(self.method),
            "n_ensemble": int(self.n_ensemble),
            "shape": [height, width],
            "mean_std": float(self.mean_std),
            "p95_std": float(self.p95_std),
            "max_std": float(self.max_std),
            "mean_novelty": float(self.mean_novelty),
            "p95_novelty": float(self.p95_novelty),
            "mean_confidence": float(_safe_mean(self.confidence_map, default=0.0)),
            "low_confidence_fraction": float(self.low_confidence_fraction()),
            "reliability_score": float(self.reliability_score),
            "hallucination_risk": str(self.hallucination_risk),
            "weights": {"std": float(weights[0]), "novelty": float(weights[1])},
            "interpretation": str(self.interpretation),
            "histogram": self.histogram(),
        }

    def low_confidence_fraction(self, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> float:
        """Fraction of finite pixels below `threshold` confidence, in [0, 1].

        Parameters
        ----------
        threshold : float, default=0.5
            Confidence below which a pixel counts as largely model-invented.

        Returns
        -------
        float
            Share of the scene the analyst should treat as unverified. 0.0 when the
            confidence map holds no finite value.
        """
        finite = _finite(self.confidence_map)
        if not finite.size:
            return 0.0
        return float(np.count_nonzero(finite < float(threshold)) / finite.size)


# --------------------------------------------------------------------------- #
# Top-level estimation
# --------------------------------------------------------------------------- #


def estimate_uncertainty(
    lr_tensor: ArrayLike,
    sr_tensor: ArrayLike,
    predict_fn: Optional[PredictFn] = None,
    n_ensemble: int = 4,
    factor: int = 4,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> UncertaintyResult:
    """Quantify which parts of a super-resolved scene are observed and which are inferred.

    With a `predict_fn` this runs a D4 test-time-augmentation ensemble and uses the
    per-pixel spread across its members as the primary uncertainty, fused with the
    novelty map. Without one it falls back to the novelty map alone, which costs no
    extra forward pass but cannot see how unstable the reconstruction is and is
    therefore optimistic.

    Both terms are normalized by their own 99th percentile (making the scale
    scene-relative and robust to a handful of extreme pixels), clipped to [0, 1] and
    combined as::

        confidence = clip(1 - (W_STD * norm(std) + W_NOVELTY * norm(novelty)), 0, 1)

    with ``W_STD = 0.6`` and ``W_NOVELTY = 0.4`` when an ensemble was run -- ensemble
    disagreement probes the model directly and is the stronger evidence -- and
    ``0.0`` / ``1.0`` for the novelty-only fallback. A term whose 99th percentile
    falls below one Sentinel-2 quantisation step (``NEGLIGIBLE_MAGNITUDE``) is
    treated as identically zero rather than rescaled, so a reconstruction that adds
    nothing to the observation scores as fully trustworthy instead of having its
    round-off stretched across the whole confidence range.

    Parameters
    ----------
    lr_tensor : np.ndarray or torch.Tensor
        Observed low-resolution scene, (C, H, W), reflectance in [0, 1].
    sr_tensor : np.ndarray or torch.Tensor
        Super-resolved scene, (C, factor*H, factor*W).
    predict_fn : Callable, optional
        Super-resolution callable, (C, H, W) -> (C, factor*H, factor*W). When None,
        the ensemble is skipped and `method` becomes "novelty-only".
    n_ensemble : int, default=4
        Requested number of D4 ensemble members, clamped to [1, 8].
    factor : int, default=4
        Nominal upscaling factor, forwarded to :func:`novelty_map`.
    progress_callback : Callable, optional
        ``progress_callback(message, percent)``, forwarded to :func:`tta_ensemble`.

    Returns
    -------
    UncertaintyResult
        Maps, scene scalars, reliability score, hallucination-risk band and the
        analyst-facing interpretation. Degenerate input (flat, empty or entirely
        non-finite) yields finite values rather than an exception.
    """
    lr = _as_chw_tensor(lr_tensor, name="lr_tensor")
    sr = _as_chw_tensor(sr_tensor, name="sr_tensor")
    target_h, target_w = int(sr.shape[-2]), int(sr.shape[-1])

    novelty = novelty_map(lr, sr, factor=factor)

    band_std_arr: np.ndarray | None = None
    if predict_fn is not None:
        members = tta_ensemble(
            lr,
            predict_fn,
            n_members=n_ensemble,
            progress_callback=progress_callback,
        )
        n_members = int(members.shape[0])
        # Population standard deviation over the member axis, per band, so that a
        # single-member ensemble degrades to exactly zero spread instead of NaN.
        band_std = torch.nan_to_num(members, nan=0.0, posinf=0.0, neginf=0.0).std(
            dim=0, correction=0
        )
        if tuple(band_std.shape[-2:]) != (target_h, target_w):
            band_std = F.interpolate(
                band_std.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )[0]
        band_std_arr = band_std.numpy().astype(np.float32, copy=False)
        std_map = band_std_arr.mean(axis=0).astype(np.float32, copy=False)
        method = "tta-d4-ensemble+novelty"
        weight_std, weight_novelty = W_STD, W_NOVELTY
    else:
        n_members = 1
        std_map = np.zeros((target_h, target_w), dtype=np.float32)
        method = "novelty-only"
        weight_std, weight_novelty = W_STD_NOVELTY_ONLY, W_NOVELTY_ONLY

    penalty = weight_std * _normalize_by_percentile(std_map)
    penalty = penalty + weight_novelty * _normalize_by_percentile(novelty)
    confidence_map = np.clip(1.0 - penalty, 0.0, 1.0).astype(np.float32)

    reliability_score = 100.0 * _safe_mean(confidence_map, default=0.0)
    risk = _risk_band(reliability_score)

    # Share of the SR grid that is finite across every band. A fully masked scene
    # would otherwise be reported as flawless simply because there is nothing in it
    # for the network to have invented.
    valid_fraction = (
        float(torch.isfinite(sr).all(dim=0).to(torch.float32).mean().item())
        if sr.numel()
        else 0.0
    )

    finite_confidence = _finite(confidence_map)
    if finite_confidence.size:
        low_pct = 100.0 * float(
            np.count_nonzero(finite_confidence < LOW_CONFIDENCE_THRESHOLD) / finite_confidence.size
        )
    else:
        low_pct = float("nan")

    return UncertaintyResult(
        std_map=std_map,
        band_std=band_std_arr,
        novelty_map=novelty,
        confidence_map=confidence_map,
        mean_std=_safe_mean(std_map, default=0.0),
        p95_std=_safe_percentile(std_map, 95.0, default=0.0),
        max_std=_safe_max(std_map, default=0.0),
        mean_novelty=_safe_mean(novelty, default=0.0),
        p95_novelty=_safe_percentile(novelty, 95.0, default=0.0),
        reliability_score=float(reliability_score),
        hallucination_risk=risk,
        n_ensemble=n_members,
        method=method,
        interpretation=_build_interpretation(
            reliability_score, risk, method, n_members, low_pct, valid_fraction
        ),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

# kind -> (attribute, colormap, vmin, fixed vmax or None for scene-relative)
_RENDER_SPECS: dict[str, tuple[str, str, float, Optional[float]]] = {
    "confidence": ("confidence_map", "rdylgn", 0.0, 1.0),
    "std": ("std_map", "inferno", 0.0, None),
    "novelty": ("novelty_map", "inferno", 0.0, None),
}


def render_uncertainty_png(
    result: UncertaintyResult,
    path: Union[str, Path],
    kind: str = "confidence",
) -> Path:
    """Save one uncertainty map as an RGBA PNG overlay for the map viewer.

    The confidence map uses a red-yellow-green ramp on a fixed [0, 1] scale, so green
    always means trustworthy and red always means largely invented and two scenes can
    be compared directly. The spread and novelty maps use `inferno` on a
    scene-relative 0 to 99th-percentile scale, so the brightest pixels are the ones
    the analyst should inspect first. Invalid pixels are written fully transparent
    and disappear when the PNG is draped over the imagery.

    Parameters
    ----------
    result : UncertaintyResult
        Output of :func:`estimate_uncertainty`.
    path : str or Path
        Destination PNG path. Parent directories are created if missing.
    kind : {"confidence", "std", "novelty"}, default="confidence"
        Which map to render.

    Returns
    -------
    Path
        Path to the written PNG.

    Raises
    ------
    ValueError
        If `kind` is not one of the three supported maps.
    """
    if kind not in _RENDER_SPECS:
        raise ValueError(
            f"Unknown uncertainty map kind {kind!r}; expected one of {sorted(_RENDER_SPECS)}"
        )

    # Imported here so this module never fails to load, and stays cheap to import,
    # when only the numeric API is needed.
    from ntro_srm.evaluation import colormaps as _cmaps

    attribute, cmap_name, vmin, vmax = _RENDER_SPECS[kind]
    array = np.asarray(getattr(result, attribute), dtype=np.float32)

    if vmax is None:
        vmax = _safe_percentile(array, NORMALIZATION_PERCENTILE, default=0.0)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    rgba = _cmaps.apply_colormap_rgba(array, name=cmap_name, vmin=float(vmin), vmax=float(vmax))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), mode="RGBA").save(out_path)
    return out_path
