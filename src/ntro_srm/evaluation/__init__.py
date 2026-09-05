"""Evaluation metrics, spectral validation, and XAI for NTRO-SRM.

Public surface, grouped by purpose:

* **Full-reference metrics** — :func:`psnr`, :func:`ssim`, :func:`sam`, :func:`ergas`,
  :func:`uiqi`, :func:`scc` and :func:`compute_all`, with presentation thresholds in
  :data:`METRIC_META`.
* **Reference-free validation** — :func:`wald_protocol_validate` scores a synthesis
  experiment against the real observation, and :func:`consistency_check` verifies that the
  super-resolved product collapses back onto the observed 10 m reflectance.
* **Uncertainty** — :func:`estimate_uncertainty` separates observed structure from detail
  the network inferred, which the problem statement requires to be stated explicitly.
* **Thematic products** — :data:`INDEX_REGISTRY` and :func:`compute_index` derive crop,
  water, built-up and burn indices on the super-resolved grid.
* **Reporting** — :func:`build_report` and :func:`save_report` produce the downloadable
  QA sheet that accompanies every product.

Importing this package is cheap: no model weights are loaded and no network access occurs.
"""

from __future__ import annotations

from ntro_srm.evaluation._common import safe_stats, to_chw_numpy, valid_mask
from ntro_srm.evaluation.colormaps import (
    COLORMAP_STOPS,
    apply_colormap,
    apply_colormap_rgba,
    build_lut,
    colormap_stops_hex,
    get_colormap,
)
from ntro_srm.evaluation.consistency import (
    ConsistencyResult,
    WaldValidationResult,
    consistency_check,
    mtf_degrade,
    spectral_fidelity,
    upsample_bicubic,
    wald_protocol_validate,
)
from ntro_srm.evaluation.indices import (
    INDEX_REGISTRY,
    SpectralIndexSpec,
    compute_index,
    compute_indices,
    index_delta_statistics,
    index_statistics,
    registry_as_json,
    render_index_png,
)
from ntro_srm.evaluation.metrics import (
    METRIC_META,
    band_psnr,
    band_rmse,
    band_ssim,
    compute_all,
    correlation,
    ergas,
    mae,
    psnr,
    q_index_per_band,
    rmse,
    sam,
    sam_map,
    scc,
    ssim,
    uiqi,
)
from ntro_srm.evaluation.report import (
    RECONSTRUCTION_CAVEAT,
    QualityReport,
    build_report,
    save_report,
)
from ntro_srm.evaluation.uncertainty import (
    UncertaintyResult,
    d4_transforms,
    estimate_uncertainty,
    novelty_map,
    render_uncertainty_png,
    tta_ensemble,
)

__all__ = [
    # Shared helpers
    "to_chw_numpy",
    "valid_mask",
    "safe_stats",
    # Colormaps
    "COLORMAP_STOPS",
    "build_lut",
    "get_colormap",
    "apply_colormap",
    "apply_colormap_rgba",
    "colormap_stops_hex",
    # Metrics
    "METRIC_META",
    "psnr",
    "band_psnr",
    "rmse",
    "band_rmse",
    "mae",
    "ssim",
    "band_ssim",
    "sam",
    "sam_map",
    "ergas",
    "uiqi",
    "q_index_per_band",
    "scc",
    "correlation",
    "compute_all",
    # Validation
    "mtf_degrade",
    "upsample_bicubic",
    "WaldValidationResult",
    "wald_protocol_validate",
    "ConsistencyResult",
    "consistency_check",
    "spectral_fidelity",
    # Uncertainty
    "UncertaintyResult",
    "d4_transforms",
    "tta_ensemble",
    "novelty_map",
    "estimate_uncertainty",
    "render_uncertainty_png",
    # Thematic products
    "SpectralIndexSpec",
    "INDEX_REGISTRY",
    "compute_index",
    "compute_indices",
    "index_statistics",
    "index_delta_statistics",
    "render_index_png",
    "registry_as_json",
    # Reporting
    "QualityReport",
    "RECONSTRUCTION_CAVEAT",
    "build_report",
    "save_report",
]
