"""Reference-free scientific validation of Sentinel-2 super-resolution (Wald's protocol).

Super-resolving Sentinel-2 from 10 m to 2.5 m creates an awkward verification problem:
no 2.5 m Sentinel-2 image exists anywhere to compare the output against, and buying
commercial very-high-resolution imagery for every scene is neither affordable nor
operationally repeatable. The remote-sensing community solved this in the pan-sharpening
literature with *Wald's protocol* (Wald, Ranchin & Mangolini 1997; Ranchin & Wald 2000;
Thomas & Wald 2005), which promotes the observed image itself to ground truth. Two
properties are asserted about any resolution-enhanced product:

**1. Consistency property.** Degrading the super-resolved product back to the original
observation scale must reproduce the observed image. This is a *necessary* condition: a
product that fails it has invented or destroyed radiometric energy that the instrument
actually measured, which would corrupt every downstream reflectance-based product
(NDVI, NDWI, NDBI, change detection). It costs no extra forward pass and can therefore
run on every job. Implemented by :func:`consistency_check`.

**2. Synthesis property.** The product must be as close as possible to the image the
sensor *would* have acquired at the finer resolution. That image does not exist, so the
property is tested one scale step down: the observed 10 m image is degraded to 40 m, the
model super-resolves 40 m -> 10 m, and the reconstruction is scored against the observed
10 m image, which is a genuine, physically measured reference. Under the scale-invariance
assumption that underpins the protocol, the accuracy measured across the 40 m -> 10 m
step is taken as an estimate of the accuracy of the operational 10 m -> 2.5 m step.
Implemented by :func:`wald_protocol_validate`, which additionally scores a plain bicubic
upsampling of the same degraded input so that the share of the improvement genuinely
attributable to the network can be reported instead of the raw metric value (a high PSNR
on a smooth scene says more about the scene than about the model).

The degradation operator matters as much as the metrics. A naive ``resize`` is not how a
satellite sees the world: the instrument's Modulation Transfer Function (MTF) attenuates
scene contrast with increasing spatial frequency before the detector integrates the
incoming radiance over its finite ground footprint. :func:`mtf_degrade` reproduces both
stages, a Gaussian point-spread function whose standard deviation scales with the
resolution ratio (the classical ``sigma = sigma_scale * ratio`` approximation used to
build MTF-matched filters, e.g. Aiazzi et al. 2006) followed by area-average pooling over
the coarse pixel footprint. The same operator is used in both directions so both
properties are tested under one, explicitly stated, sensor assumption.

Everything here is CPU-only, deterministic, free of randomness, and uses nothing beyond
``numpy`` and ``torch.nn.functional``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ntro_srm.evaluation import metrics as _metrics
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES, handle_nans

__all__ = [
    "PredictFn",
    "WaldValidationResult",
    "ConsistencyResult",
    "mtf_degrade",
    "upsample_bicubic",
    "improvement",
    "wald_protocol_validate",
    "consistency_check",
    "spectral_fidelity",
]

# A super-resolution callable: (C, H, W) float reflectance in [0, 1] -> (C, fH, fW).
PredictFn = Callable[[torch.Tensor], torch.Tensor]

ArrayLike = np.ndarray | torch.Tensor

# Keys emitted by metrics.compute_all that describe the tensor rather than its quality.
# They are never differenced when reporting an improvement over the baseline.
_STRUCTURAL_KEYS: frozenset[str] = frozenset({"n_bands", "shape", "per_band", "band_names"})


# --------------------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------------------


def _as_chw_tensor(x: ArrayLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Coerce an array or tensor to a detached CPU (C, H, W) float tensor.

    Accepts (H, W) -> (1, H, W), (C, H, W), and (1, C, H, W) -> (C, H, W).
    """
    if isinstance(x, torch.Tensor):
        tensor = x.detach().to(device="cpu", dtype=dtype)
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(x), dtype=np.float32)).to(dtype)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Batched input is not supported, got shape {tuple(tensor.shape)}")
        tensor = tensor.squeeze(0)
    elif tensor.ndim != 3:
        raise ValueError(f"Expected a 2D, 3D or 4D array, got shape {tuple(tensor.shape)}")

    return tensor


def _sanitize(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with 0.0, but only when a non-finite sample is actually present.

    Convolution and interpolation spread a single non-finite sample across their whole
    kernel footprint, so degradation is always performed on a sanitized copy. The
    all-finite fast path avoids a pointless full-tensor copy.
    """
    if bool(torch.isfinite(tensor).all()):
        return tensor
    return handle_nans(tensor, fill_value=0.0)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detached, C-contiguous numpy view of a CPU tensor."""
    return tensor.detach().cpu().contiguous().numpy()


def _resolve_band_names(n_bands: int, band_names: Sequence[str] | None = None) -> list[str]:
    """Name every band, defaulting to the canonical Sentinel-2 10-band ordering."""
    if band_names is not None:
        names = [str(name) for name in band_names]
        if len(names) >= n_bands:
            return names[:n_bands]
        return names + [f"Band {i + 1}" for i in range(len(names), n_bands)]
    if n_bands == len(S2_10BAND_NAMES):
        return list(S2_10BAND_NAMES)
    return [f"Band {i + 1}" for i in range(n_bands)]


def _finite_mean(arr: np.ndarray) -> float:
    """Mean over finite samples; NaN when the array holds no finite sample."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _as_float(value: Any) -> float | None:
    """Return `value` as a python float, or None when it is not a numeric scalar."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and containers to pure-python values."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _jsonify(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(key): _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(value) for value in obj]
    return obj


def _gaussian_kernel1d(sigma: float, radius: int, dtype: torch.dtype) -> torch.Tensor:
    """Unit-sum 1-D Gaussian of length ``2 * radius + 1``, accumulated in float64."""
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    return kernel.to(dtype)


def _gaussian_blur_separable(
    tensor: torch.Tensor,
    sigma: float,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Depthwise separable Gaussian blur of a (C, H, W) tensor with reflect padding.

    A 2-D isotropic Gaussian is separable, so the (2r+1)^2 convolution is applied as two
    1-D passes costing O(C * H * W * r) instead of O(C * H * W * r^2). ``groups=C`` keeps
    every spectral band independent (no cross-band leakage), and 'reflect' padding avoids
    the dark halo that zero padding would introduce along the scene border.

    The radius is clamped to ``H - 1`` / ``W - 1`` because ``F.pad(mode='reflect')`` cannot
    mirror more samples than the axis contains; a degenerate radius of zero skips the blur.
    """
    n_bands, height, width = tensor.shape
    radius = int(math.ceil(float(truncate) * float(sigma)))
    radius = min(radius, height - 1, width - 1)
    if sigma <= 0.0 or radius < 1:
        return tensor

    kernel = _gaussian_kernel1d(sigma, radius, tensor.dtype)
    kernel_x = kernel.view(1, 1, 1, -1).repeat(n_bands, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(n_bands, 1, 1, 1)

    batched = tensor.unsqueeze(0)
    batched = F.pad(batched, (radius, radius, 0, 0), mode="reflect")
    batched = F.conv2d(batched, kernel_x, groups=n_bands)
    batched = F.pad(batched, (0, 0, radius, radius), mode="reflect")
    batched = F.conv2d(batched, kernel_y, groups=n_bands)
    return batched.squeeze(0)


# --------------------------------------------------------------------------------------
# Sensor-like degradation and interpolation
# --------------------------------------------------------------------------------------


def mtf_degrade(
    tensor: ArrayLike,
    factor: int = 4,
    sigma_scale: float = 0.5,
    antialias: bool = True,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Degrade an image the way a coarser-resolution sensor would observe the same scene.

    Two physical stages are modelled, in the order light actually encounters them:

    1. **Optical MTF.** The instrument's point-spread function low-pass filters the scene
       before it is sampled. It is approximated by an isotropic Gaussian with
       ``sigma = sigma_scale * factor``, expressed in *fine-grid* pixels. The default
       ``sigma_scale = 0.5`` gives ``sigma = 2`` px at a 4x ratio, a full width at half
       maximum of ~4.7 fine pixels, so the response is roughly halved at the coarse
       Nyquist frequency. That is far closer to the MTF-at-Nyquist figures published for
       the Sentinel-2 MSI (~0.15-0.30) than an ideal box or a plain resize, and it is the
       ``sigma = ratio / 2`` heuristic used to build MTF-matched pan-sharpening filters.
    2. **Detector integration.** The coarse detector averages the radiance falling inside
       its ground footprint, which is exactly ``F.avg_pool2d`` with a ``factor``-wide
       non-overlapping window. Area averaging, rather than point sampling, is what makes
       the operator radiometry-preserving: the mean reflectance of the degraded image
       equals the mean reflectance of its input, the property that
       :func:`consistency_check` relies on to attribute any residual to the model.

    Disabling the blur (``antialias=False``) leaves the pure box-average operator, which
    aliases high-frequency texture and is only useful for ablation studies.

    Parameters
    ----------
    tensor : np.ndarray or torch.Tensor
        Image of shape (C, H, W), (H, W) or (1, C, H, W), float reflectance in [0, 1].
        Non-finite samples are replaced with 0.0 first so a single bad pixel cannot smear
        across its whole kernel footprint.
    factor : int, default=4
        Integer resolution ratio. ``factor=1`` applies the blur only.
    sigma_scale : float, default=0.5
        Gaussian sigma per unit of `factor`, in fine-grid pixels.
    antialias : bool, default=True
        Apply the MTF blur before pooling. Leave enabled for any scientific use.
    truncate : float, default=3.0
        Gaussian kernel half-width in standard deviations. The resulting radius is
        additionally clamped so that reflect padding stays legal on small tiles.

    Returns
    -------
    torch.Tensor
        Degraded image of shape (C, H // factor, W // factor), float32.

    Notes
    -----
    The input is cropped to the largest multiple of `factor` before pooling, discarding at
    most ``factor - 1`` rows from the bottom edge and ``factor - 1`` columns from the right
    edge. Cropping rather than padding guarantees that every output pixel integrates a
    complete ``factor x factor`` footprint, so no border pixel carries a partial, and
    therefore radiometrically biased, average. Callers that need the two grids to line up
    must crop their own reference identically -- :func:`wald_protocol_validate` and
    :func:`consistency_check` do that for you.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor}")

    image = _sanitize(_as_chw_tensor(tensor))
    _, height, width = image.shape

    cropped_h = (height // factor) * factor
    cropped_w = (width // factor) * factor
    if cropped_h < factor or cropped_w < factor:
        raise ValueError(
            f"Image of size {height}x{width} is too small to degrade by a factor of {factor}"
        )
    if (cropped_h, cropped_w) != (height, width):
        image = image[:, :cropped_h, :cropped_w]

    if antialias:
        sigma = float(sigma_scale) * float(factor)
        image = _gaussian_blur_separable(image, sigma, truncate=truncate)

    if factor == 1:
        return image.contiguous()

    pooled = F.avg_pool2d(image.unsqueeze(0), kernel_size=factor, stride=factor)
    return pooled.squeeze(0)


def upsample_bicubic(
    tensor: ArrayLike,
    factor: int = 4,
    size: int | tuple[int, int] | None = None,
    clamp: tuple[float, float] | None = (0.0, 1.0),
) -> torch.Tensor:
    """Bicubic interpolation: the classical baseline that any SR model must beat.

    Bicubic interpolation adds no information, it only redistributes the observed samples
    onto a finer grid. Scoring it against the same reference as the network is what turns
    an absolute metric into an attributable one: any gain over this baseline is detail the
    model contributed, not detail the scene happened to contain.

    Parameters
    ----------
    tensor : np.ndarray or torch.Tensor
        Image of shape (C, H, W), (H, W) or (1, C, H, W).
    factor : int, default=4
        Integer magnification. Ignored when `size` is given.
    size : int or tuple of int, optional
        Explicit output (H, W); use it to match an existing grid exactly.
    clamp : tuple of float, optional
        Range the result is clipped to, ``(0.0, 1.0)`` by default. Cubic kernels overshoot
        at strong edges and can produce physically impossible negative reflectance. Pass
        ``None`` to keep the raw interpolation.

    Returns
    -------
    torch.Tensor
        Interpolated image of shape (C, factor * H, factor * W), or (C, *size), float32.
    """
    image = _sanitize(_as_chw_tensor(tensor))

    if size is None:
        factor = int(factor)
        if factor < 1:
            raise ValueError(f"factor must be a positive integer, got {factor}")
        upsampled = F.interpolate(
            image.unsqueeze(0),
            scale_factor=float(factor),
            mode="bicubic",
            align_corners=False,
        )
    else:
        target = (int(size), int(size)) if isinstance(size, int) else tuple(int(s) for s in size)
        upsampled = F.interpolate(
            image.unsqueeze(0),
            size=target,
            mode="bicubic",
            align_corners=False,
        )

    result = upsampled.squeeze(0)
    if clamp is not None:
        result = result.clamp(min=float(clamp[0]), max=float(clamp[1]))
    return result


# --------------------------------------------------------------------------------------
# Improvement over the interpolation baseline
# --------------------------------------------------------------------------------------


def improvement(model_metrics: dict, baseline_metrics: dict) -> dict[str, float]:
    """Signed gain of a model over a baseline, oriented so that positive always means better.

    Quality metrics disagree about which direction is good: PSNR, SSIM, UIQI and the
    correlation coefficients should go up, while RMSE, MAE, SAM and ERGAS should go down.
    Each shared numeric key is therefore differenced according to
    ``metrics.METRIC_META[key]['better']``::

        'higher' -> gain = model - baseline
        'lower'  -> gain = baseline - model

    so a positive number is an unambiguous win for the model in every row of the table the
    web UI renders, and the reader never has to remember the polarity of nine metrics.
    Keys that merely describe the tensor (``n_bands``, ``shape``, ``per_band``) and keys
    whose value is not a numeric scalar are skipped; a key absent from ``METRIC_META`` is
    assumed to be 'higher is better'.

    Parameters
    ----------
    model_metrics : dict
        Output of ``metrics.compute_all`` for the model reconstruction.
    baseline_metrics : dict
        Output of ``metrics.compute_all`` for the bicubic reconstruction.

    Returns
    -------
    dict of str to float
        One signed gain per shared numeric metric key. NaN propagates when either side is
        undefined.
    """
    meta = getattr(_metrics, "METRIC_META", None) or {}
    gains: dict[str, float] = {}

    for key, model_value in model_metrics.items():
        if key not in baseline_metrics:
            continue
        if key in _STRUCTURAL_KEYS and key not in meta:
            continue
        model_scalar = _as_float(model_value)
        baseline_scalar = _as_float(baseline_metrics[key])
        if model_scalar is None or baseline_scalar is None:
            continue
        entry = meta.get(key) or {}
        better = str(entry.get("better", "higher")).lower()
        if better == "lower":
            gains[key] = float(baseline_scalar - model_scalar)
        else:
            gains[key] = float(model_scalar - baseline_scalar)

    return gains


# --------------------------------------------------------------------------------------
# Property 2 - synthesis (Wald's protocol)
# --------------------------------------------------------------------------------------


@dataclass
class WaldValidationResult:
    """Accuracy of a super-resolution model measured with no external reference imagery,
    together with the interpolation baseline it has to beat.

    Attributes
    ----------
    metrics : dict
        ``metrics.compute_all`` of the model reconstruction against the observed image.
    baseline_metrics : dict
        The same metrics for a bicubic upsampling of the identical degraded input.
    improvement : dict
        Per-metric signed gain of the model over the baseline; positive is always better.
    reference_shape : list of int
        (C, H, W) of the observed image actually used as ground truth, after cropping.
    degraded_shape : list of int
        (C, H, W) of the synthetic coarse-resolution input handed to the model.
    scale_factor : int
        Resolution ratio between the two grids.
    protocol : str
        Human-readable description of the scale chain that was exercised.
    band_names : list of str
        Names of the spectral bands, in stack order.
    """

    metrics: dict
    baseline_metrics: dict
    improvement: dict
    reference_shape: list[int]
    degraded_shape: list[int]
    scale_factor: int
    protocol: str
    band_names: list[str]

    def to_dict(self) -> dict:
        """JSON-serialisable view with every numpy scalar cast to a python number."""
        return {
            "protocol": str(self.protocol),
            "scale_factor": int(self.scale_factor),
            "reference_shape": [int(v) for v in self.reference_shape],
            "degraded_shape": [int(v) for v in self.degraded_shape],
            "band_names": [str(name) for name in self.band_names],
            "metrics": _jsonify(self.metrics),
            "baseline_metrics": _jsonify(self.baseline_metrics),
            "improvement": {str(k): float(v) for k, v in dict(self.improvement).items()},
        }


def wald_protocol_validate(
    lr_tensor: ArrayLike,
    predict_fn: PredictFn,
    factor: int = 4,
    band_names: Sequence[str] | None = None,
) -> WaldValidationResult:
    """Score a super-resolution model with Wald's synthesis protocol.

    The observed Sentinel-2 image is promoted to ground truth and the whole experiment is
    shifted one scale step down, to where a real reference exists:

    1. degrade the observed 10 m image to a synthetic 40 m image with :func:`mtf_degrade`
       (Gaussian MTF blur followed by detector-footprint averaging);
    2. super-resolve that 40 m image back to 10 m with `predict_fn` -- exactly the
       operation the model performs operationally, only one octave coarser;
    3. score the reconstruction against the *original observed* 10 m image using
       ``metrics.compute_all``.

    The same degraded input is also upsampled bicubically and scored, so the report can
    separate "this scene is easy" from "this model is good": a 38 dB PSNR over farmland
    says almost nothing on its own, because bicubic would also score 37 dB there. Only the
    difference between the two columns is attributable to the network, and it is returned
    sign-corrected in :attr:`WaldValidationResult.improvement`.

    The figures produced are genuine quantitative accuracy measurements obtained with no
    commercial imagery and no field campaign. Their transfer to the operational
    10 m -> 2.5 m step rests on the scale-invariance assumption of Wald's protocol -- that
    the model's behaviour does not change qualitatively with absolute scale -- which is the
    standard, and standardly stated, caveat of the method.

    Parameters
    ----------
    lr_tensor : np.ndarray or torch.Tensor
        The observed image, shape (C, H, W), float reflectance in [0, 1]. Despite the
        parameter name it plays the role of the high-resolution reference here.
    predict_fn : PredictFn
        Callable mapping a (C, h, w) tensor to (C, factor * h, factor * w). It is invoked
        exactly once, inside ``torch.no_grad``.
    factor : int, default=4
        Resolution ratio used for both the degradation and the reconstruction.
    band_names : sequence of str, optional
        Band labels; defaults to the canonical Sentinel-2 10-band ordering when C == 10.

    Returns
    -------
    WaldValidationResult
        Model metrics, baseline metrics, and the improvement attributable to the model.

    Notes
    -----
    The reference is cropped to the exact spatial extent returned by `predict_fn` (and by
    the bicubic baseline) before scoring, so all three grids are pixel-aligned and no
    metric is ever computed across a shape mismatch. A model that returns slightly less
    than ``factor`` times the degraded size simply has the comparison area trimmed.
    """
    factor = int(factor)
    reference = _sanitize(_as_chw_tensor(lr_tensor))

    degraded = mtf_degrade(reference, factor=factor)

    with torch.no_grad():
        model_sr = _as_chw_tensor(predict_fn(degraded))
    baseline_sr = upsample_bicubic(degraded, factor=factor)

    n_bands = reference.shape[0]
    if model_sr.shape[0] != n_bands:
        raise ValueError(
            f"predict_fn returned {model_sr.shape[0]} bands but the reference has {n_bands}"
        )

    # Align all three grids: the reference is cropped to the reconstructed extent.
    height = min(reference.shape[1], model_sr.shape[1], baseline_sr.shape[1])
    width = min(reference.shape[2], model_sr.shape[2], baseline_sr.shape[2])
    reference = reference[:, :height, :width]
    model_sr = _sanitize(model_sr[:, :height, :width])
    baseline_sr = baseline_sr[:, :height, :width]

    names = _resolve_band_names(n_bands, band_names)
    reference_np = _to_numpy(reference)

    model_metrics = _metrics.compute_all(
        _to_numpy(model_sr),
        reference_np,
        data_range=1.0,
        ratio=float(factor),
        band_names=names,
    )
    baseline_metrics = _metrics.compute_all(
        _to_numpy(baseline_sr),
        reference_np,
        data_range=1.0,
        ratio=float(factor),
        band_names=names,
    )

    return WaldValidationResult(
        metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        improvement=improvement(model_metrics, baseline_metrics),
        reference_shape=[int(v) for v in reference.shape],
        degraded_shape=[int(v) for v in degraded.shape],
        scale_factor=factor,
        protocol=f"Wald synthesis (10 m -> {10 * factor:g} m -> 10 m)",
        band_names=names,
    )


# --------------------------------------------------------------------------------------
# Property 1 - consistency
# --------------------------------------------------------------------------------------


@dataclass
class ConsistencyResult:
    """Whether a super-resolved product still agrees with what Sentinel-2 measured.

    Attributes
    ----------
    metrics : dict
        ``metrics.compute_all`` of the down-degraded product against the observed image.
    per_band_bias : list of float
        Signed mean of ``sr_down - lr`` per band, in reflectance units. The sign matters:
        a positive value means the product systematically brightens that band.
    per_band_rmse : list of float
        Root-mean-square difference per band, in reflectance units.
    max_abs_bias : float
        Largest absolute per-band bias; the headline consistency number.
    spectral_angle_deg : float
        Mean spectral angle between the down-degraded and observed spectra, in degrees.
        NaN when every pixel spectrum is degenerate (e.g. an all-zero tile).
    passed : bool
        True when both tolerances are met.
    tolerance : dict
        The thresholds that were applied.
    band_names : list of str
        Names of the spectral bands, in stack order.
    """

    metrics: dict
    per_band_bias: list[float]
    per_band_rmse: list[float]
    max_abs_bias: float
    spectral_angle_deg: float
    passed: bool
    tolerance: dict
    band_names: list[str]

    def to_dict(self) -> dict:
        """JSON-serialisable view with every numpy scalar cast to a python number."""
        return {
            "passed": bool(self.passed),
            "max_abs_bias": float(self.max_abs_bias),
            "spectral_angle_deg": float(self.spectral_angle_deg),
            "per_band_bias": [float(v) for v in self.per_band_bias],
            "per_band_rmse": [float(v) for v in self.per_band_rmse],
            "tolerance": {str(k): float(v) for k, v in dict(self.tolerance).items()},
            "band_names": [str(name) for name in self.band_names],
            "metrics": _jsonify(self.metrics),
        }


def consistency_check(
    lr_tensor: ArrayLike,
    sr_tensor: ArrayLike,
    factor: int = 4,
    band_names: Sequence[str] | None = None,
    max_abs_bias: float = 0.01,
    max_sam_deg: float = 1.0,
) -> ConsistencyResult:
    """Test Wald's consistency property on an already-computed super-resolved product.

    Degrading the 2.5 m product back onto the native 10 m grid -- with the *same* MTF plus
    detector-averaging operator used everywhere else in this module -- must reproduce the
    reflectance Sentinel-2 actually measured. Because area averaging conserves radiometric
    energy, any residual is real: the model has either added or removed energy the
    instrument never saw. That is the failure mode which silently corrupts NDVI, NDWI,
    NDBI and every change-detection product computed downstream, which is why the *signed*
    per-band bias is reported next to the RMSE -- a model can look beautifully sharp and
    still be unusable if it shifts B08 by two reflectance points.

    Two complementary criteria are applied:

    * ``max_abs_bias`` bounds the *radiometric* error, per band, in reflectance units. The
      0.01 default is of the order of the radiometric uncertainty of Sentinel-2 L2A surface
      reflectance itself, so it asks the product to be consistent to within the noise of
      the observation it came from.
    * ``max_sam_deg`` bounds the *spectral* error: the mean angle between the down-degraded
      and observed spectral vectors. This catches a product whose bands are each
      individually close but whose band *ratios* have drifted -- precisely what index
      arithmetic is sensitive to, and precisely what a per-band bias check alone misses.

    The test needs no extra forward pass and no external reference, so it can run on every
    job. It is a necessary but not a sufficient condition: a product can be perfectly
    consistent and still hallucinate texture *within* each coarse pixel, which is what
    :func:`wald_protocol_validate` and the uncertainty module exist to probe.

    Parameters
    ----------
    lr_tensor : np.ndarray or torch.Tensor
        The observed native-resolution image, shape (C, H, W).
    sr_tensor : np.ndarray or torch.Tensor
        The super-resolved product, shape (C, factor * H, factor * W).
    factor : int, default=4
        Resolution ratio between the two grids.
    band_names : sequence of str, optional
        Band labels; defaults to the canonical Sentinel-2 10-band ordering when C == 10.
    max_abs_bias : float, default=0.01
        Tolerance on the largest absolute per-band mean difference, in reflectance units.
    max_sam_deg : float, default=1.0
        Tolerance on the mean spectral angle, in degrees.

    Returns
    -------
    ConsistencyResult
        Per-band bias and RMSE, the full metric set, and the pass/fail verdict.

    Notes
    -----
    The observed image is cropped to the extent recovered from the product, so a product
    whose size is not exactly ``factor`` times the input is still evaluated over the region
    the two grids share. A non-finite spectral angle -- undefined when every pixel spectrum
    is the zero vector -- is treated as "no evidence of failure" rather than as a failure,
    whereas a non-finite bias always fails.
    """
    factor = int(factor)
    observed = _sanitize(_as_chw_tensor(lr_tensor))
    product = _sanitize(_as_chw_tensor(sr_tensor))

    n_bands = observed.shape[0]
    if product.shape[0] != n_bands:
        raise ValueError(
            f"The product has {product.shape[0]} bands but the observation has {n_bands}"
        )

    # Bring the product back to the observation grid under the same sensor assumption.
    product_down = mtf_degrade(product, factor=factor)

    height = min(observed.shape[1], product_down.shape[1])
    width = min(observed.shape[2], product_down.shape[2])
    observed = observed[:, :height, :width]
    product_down = product_down[:, :height, :width]

    names = _resolve_band_names(n_bands, band_names)

    observed_np = _to_numpy(observed)
    product_np = _to_numpy(product_down)
    residual = product_np.astype(np.float64) - observed_np.astype(np.float64)

    per_band_bias = [_finite_mean(residual[b]) for b in range(n_bands)]
    per_band_rmse: list[float] = []
    for b in range(n_bands):
        mean_square = _finite_mean(residual[b] ** 2)
        per_band_rmse.append(
            float(math.sqrt(mean_square)) if math.isfinite(mean_square) else float("nan")
        )

    # A single band whose bias is undefined makes the whole verdict undefined.
    finite_bias = [abs(v) for v in per_band_bias if math.isfinite(v)]
    if finite_bias and len(finite_bias) == n_bands:
        max_bias = float(max(finite_bias))
    else:
        max_bias = float("nan")

    metrics_dict = _metrics.compute_all(
        product_np,
        observed_np,
        data_range=1.0,
        ratio=float(factor),
        band_names=names,
    )
    sam_value = _as_float(metrics_dict.get("sam_deg"))
    spectral_angle_deg = float(sam_value) if sam_value is not None else float("nan")

    bias_ok = math.isfinite(max_bias) and max_bias <= float(max_abs_bias)
    sam_ok = (not math.isfinite(spectral_angle_deg)) or spectral_angle_deg <= float(max_sam_deg)

    return ConsistencyResult(
        metrics=metrics_dict,
        per_band_bias=[float(v) for v in per_band_bias],
        per_band_rmse=[float(v) for v in per_band_rmse],
        max_abs_bias=max_bias,
        spectral_angle_deg=spectral_angle_deg,
        passed=bool(bias_ok and sam_ok),
        tolerance={
            "max_abs_bias": float(max_abs_bias),
            "spectral_angle_deg": float(max_sam_deg),
        },
        band_names=names,
    )


# --------------------------------------------------------------------------------------
# Per-band reflectance bookkeeping
# --------------------------------------------------------------------------------------


def spectral_fidelity(
    lr_tensor: ArrayLike,
    sr_tensor: ArrayLike,
    factor: int = 4,
    band_names: Sequence[str] | None = None,
) -> dict:
    """Compare mean reflectance band by band between the observation and the product.

    Super-resolution must be radiometrically neutral: it redistributes energy inside each
    coarse pixel but must not change how much of it there is. This is the plainest possible
    statement of that requirement -- the scene-average reflectance of every band, before
    and after -- and it is what lets an analyst confirm at a glance that B08 has not
    drifted before trusting an NDVI map derived from the product.

    Unlike :func:`consistency_check` no degradation is performed here; the two grids are
    simply averaged over the same ground footprint. The relative difference is reported
    beside the absolute one because a 0.002 shift is negligible for a bright NIR band and
    substantial for a dark SWIR one.

    Parameters
    ----------
    lr_tensor : np.ndarray or torch.Tensor
        The observed native-resolution image, shape (C, H, W).
    sr_tensor : np.ndarray or torch.Tensor
        The super-resolved product, shape (C, factor * H, factor * W).
    factor : int, default=4
        Resolution ratio, used to crop both rasters to a common ground footprint so the
        two means describe identical terrain. Cropping is skipped when the product is
        smaller than one coarse pixel.
    band_names : sequence of str, optional
        Band labels; defaults to the canonical Sentinel-2 10-band ordering when C == 10.

    Returns
    -------
    dict
        ``{"bands": [{"band", "lr_mean", "sr_mean", "delta", "rel_pct"}, ...],
        "max_rel_pct": float, "mean_abs_delta": float}`` where ``delta`` is
        ``sr_mean - lr_mean`` in reflectance units, ``rel_pct`` is that delta as a
        percentage of the observed mean (NaN for a band whose observed mean is zero, where
        a relative error is undefined), ``max_rel_pct`` is the largest absolute ``rel_pct``
        over all bands, and ``mean_abs_delta`` is the mean of ``|delta|``.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor}")

    observed = _as_chw_tensor(lr_tensor)
    product = _as_chw_tensor(sr_tensor)

    n_bands = min(observed.shape[0], product.shape[0])
    observed = observed[:n_bands]
    product = product[:n_bands]

    # Restrict both rasters to the ground footprint they share.
    common_h = min(observed.shape[1], product.shape[1] // factor)
    common_w = min(observed.shape[2], product.shape[2] // factor)
    if common_h >= 1 and common_w >= 1:
        observed = observed[:, :common_h, :common_w]
        product = product[:, : common_h * factor, : common_w * factor]

    observed_np = _to_numpy(observed).astype(np.float64)
    product_np = _to_numpy(product).astype(np.float64)
    names = _resolve_band_names(n_bands, band_names)

    bands: list[dict[str, Any]] = []
    abs_deltas: list[float] = []
    rel_pcts: list[float] = []

    for index in range(n_bands):
        lr_mean = _finite_mean(observed_np[index])
        sr_mean = _finite_mean(product_np[index])
        delta = float(sr_mean - lr_mean)
        if math.isfinite(lr_mean) and abs(lr_mean) > 1e-12 and math.isfinite(delta):
            rel_pct = float(100.0 * delta / lr_mean)
        else:
            rel_pct = float("nan")

        bands.append(
            {
                "band": names[index],
                "lr_mean": float(lr_mean),
                "sr_mean": float(sr_mean),
                "delta": delta,
                "rel_pct": rel_pct,
            }
        )
        if math.isfinite(delta):
            abs_deltas.append(abs(delta))
        if math.isfinite(rel_pct):
            rel_pcts.append(abs(rel_pct))

    return {
        "bands": bands,
        "max_rel_pct": float(max(rel_pcts)) if rel_pcts else float("nan"),
        "mean_abs_delta": float(sum(abs_deltas) / len(abs_deltas)) if abs_deltas else float("nan"),
    }
