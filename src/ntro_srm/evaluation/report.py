"""Quality-assessment report assembled for every super-resolved product.

The report is the document a reviewer reads instead of taking the imagery on trust. It
states what was validated and, just as importantly, what was not: the Wald synthesis
protocol establishes accuracy at the 40 m to 10 m scale, never directly at 2.5 m against
commercial reference imagery, and that distinction is carried explicitly into every
verdict and caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ntro_srm.evaluation import metrics as _metrics

# Always stated, regardless of how well the product scored.
RECONSTRUCTION_CAVEAT: str = (
    "High-frequency spatial detail in the 2.5 m product is a neural reconstruction "
    "inferred by the model and is not directly observed by the Sentinel-2 sensor. "
    "Spectral values remain calibrated to Sentinel-2 Level-2A surface reflectance."
)

SCALE_CAVEAT: str = (
    "Accuracy figures come from Wald's synthesis protocol at the 40 m to 10 m scale. They "
    "are indicative of, but not a direct measurement of, performance at 2.5 m. Direct "
    "validation would require co-registered high-resolution reference imagery."
)

# Metrics promoted to the summary, in display order.
HEADLINE_KEYS: tuple[str, ...] = ("psnr_db", "ssim", "sam_deg", "ergas")


def _as_dict(obj: Any) -> dict[str, Any] | None:
    """Normalise a dataclass result, a plain dict, or ``None`` to a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else None
    return None


def _num(value: Any, digits: int = 4) -> str:
    """Format a number for a markdown table, tolerating ``None`` and non-finite values."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if f != f or f in (float("inf"), float("-inf")):
        return "—"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".") if digits > 0 else f"{f:.0f}"


def _rate(value: Any, meta: dict[str, Any] | None) -> str:
    """Classify a metric value as excellent / good / fair against its thresholds."""
    if meta is None:
        return "fair"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "fair"
    if f != f:
        return "fair"
    if meta.get("better") == "lower":
        if f <= meta.get("excellent", float("-inf")):
            return "excellent"
        if f <= meta.get("good", float("-inf")):
            return "good"
        return "fair"
    if f >= meta.get("excellent", float("inf")):
        return "excellent"
    if f >= meta.get("good", float("inf")):
        return "good"
    return "fair"


@dataclass
class QualityReport:
    """Structured quality-assessment record for one super-resolution job.

    Attributes
    ----------
    job_id : str
        Identifier of the job the report describes.
    generated_at : str
        ISO-8601 UTC timestamp.
    scene : dict
        Free-form job metadata (scene identifier, model, device, grid, CRS).
    wald, consistency, uncertainty, spectral_fidelity : dict or None
        Normalised results of each assessment stage.
    indices : list[dict] or None
        Per-index statistics from :func:`ntro_srm.evaluation.indices.index_statistics`.
    summary : dict
        ``verdict``, ``headline_metrics`` and ``caveats``.
    """

    job_id: str
    generated_at: str
    scene: dict[str, Any] = field(default_factory=dict)
    wald: dict[str, Any] | None = None
    consistency: dict[str, Any] | None = None
    uncertainty: dict[str, Any] | None = None
    spectral_fidelity: dict[str, Any] | None = None
    indices: list[dict[str, Any]] | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the whole report."""
        return {
            "job_id": self.job_id,
            "generated_at": self.generated_at,
            "scene": self.scene,
            "wald": self.wald,
            "consistency": self.consistency,
            "uncertainty": self.uncertainty,
            "spectral_fidelity": self.spectral_fidelity,
            "indices": self.indices,
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        """Render the report as a self-contained markdown QA sheet."""
        lines: list[str] = []
        add = lines.append

        add(f"# NTRO-SRM Quality Assessment — `{self.job_id}`")
        add("")
        add(f"_Generated {self.generated_at}_")
        add("")

        # --- Scene ---
        if self.scene:
            add("## Product")
            add("")
            add("| Property | Value |")
            add("| --- | --- |")
            for key, value in self.scene.items():
                label = str(key).replace("_", " ").capitalize()
                add(f"| {label} | `{value}` |")
            add("")

        # --- Summary ---
        verdict = self.summary.get("verdict", "inconclusive")
        add("## Verdict")
        add("")
        add(f"**{verdict.replace('-', ' ').capitalize()}**")
        add("")
        headline = self.summary.get("headline_metrics") or []
        if headline:
            add("| Metric | Value | Rating |")
            add("| --- | ---: | --- |")
            for item in headline:
                unit = f" {item.get('unit')}" if item.get("unit") else ""
                add(f"| {item.get('label')} | {_num(item.get('value'), 4)}{unit} | {item.get('rating')} |")
            add("")

        # --- Wald ---
        add("## Accuracy — Wald synthesis protocol")
        add("")
        if self.wald and self.wald.get("metrics"):
            add(
                "The observed 10 m image is degraded to 40 m with a sensor-like modulation "
                "transfer function, reconstructed back to 10 m by the same network, and scored "
                "against the original observation. A bicubic upsample of the same degraded input "
                "provides the baseline, so the gain attributable to the network is explicit."
            )
            add("")
            protocol = self.wald.get("protocol", "Wald synthesis")
            add(f"Protocol: `{protocol}` · scale factor {self.wald.get('scale_factor', 4)}")
            add("")
            add("| Metric | Model | Bicubic baseline | Gain |")
            add("| --- | ---: | ---: | ---: |")
            model_metrics = self.wald.get("metrics") or {}
            baseline = self.wald.get("baseline_metrics") or {}
            improvement = self.wald.get("improvement") or {}
            for key in ("psnr_db", "ssim", "sam_deg", "ergas", "rmse", "mae", "uiqi", "scc", "cc"):
                if key not in model_metrics:
                    continue
                meta = _metrics.METRIC_META.get(key, {})
                digits = 2 if key == "psnr_db" else 4
                add(
                    f"| {meta.get('label', key)} | {_num(model_metrics.get(key), digits)} "
                    f"| {_num(baseline.get(key), digits)} | {_num(improvement.get(key), digits)} |"
                )
            add("")

            per_band = model_metrics.get("per_band") or []
            if per_band:
                add("### Per-band reconstruction")
                add("")
                add("| Band | PSNR (dB) | SSIM | RMSE |")
                add("| --- | ---: | ---: | ---: |")
                for entry in per_band:
                    add(
                        f"| {entry.get('band')} | {_num(entry.get('psnr_db'), 2)} "
                        f"| {_num(entry.get('ssim'), 4)} | {_num(entry.get('rmse'), 5)} |"
                    )
                add("")
        else:
            add("_Not run for this product._")
            add("")

        # --- Consistency ---
        add("## Radiometric consistency")
        add("")
        if self.consistency:
            passed = bool(self.consistency.get("passed"))
            tolerance = self.consistency.get("tolerance") or {}
            add(
                "Wald's consistency property: downsampling the 2.5 m product back onto the native "
                "10 m grid must reproduce the observed Sentinel-2 reflectance."
            )
            add("")
            add(f"**{'Within tolerance' if passed else 'Out of tolerance'}**")
            add("")
            add("| Check | Measured | Tolerance |")
            add("| --- | ---: | ---: |")
            add(
                f"| Max absolute band bias | {_num(self.consistency.get('max_abs_bias'), 5)} "
                f"| {_num(tolerance.get('max_abs_bias'), 3)} |"
            )
            add(
                f"| Mean spectral angle (deg) | {_num(self.consistency.get('spectral_angle_deg'), 3)} "
                f"| {_num(tolerance.get('spectral_angle_deg'), 2)} |"
            )
            cons_metrics = self.consistency.get("metrics") or {}
            add(f"| Round-trip PSNR (dB) | {_num(cons_metrics.get('psnr_db'), 2)} | — |")
            add(f"| Round-trip SSIM | {_num(cons_metrics.get('ssim'), 4)} | — |")
            add("")
        else:
            add("_Not available._")
            add("")

        # --- Uncertainty ---
        add("## Reconstruction uncertainty")
        add("")
        if self.uncertainty:
            u = self.uncertainty
            add(f"- Reliability score: **{_num(u.get('reliability_score'), 1)} / 100**")
            add(f"- Hallucination risk band: **{u.get('hallucination_risk', '—')}**")
            add(f"- Method: `{u.get('method', '—')}` with {u.get('n_ensemble', 1)} ensemble member(s)")
            add(f"- Mean ensemble spread: {_num(u.get('mean_std'), 5)} (95th pct {_num(u.get('p95_std'), 5)})")
            add(f"- Mean synthesised detail: {_num(u.get('mean_novelty'), 5)} (95th pct {_num(u.get('p95_novelty'), 5)})")
            add("")
            if u.get("interpretation"):
                add(str(u["interpretation"]))
                add("")
        else:
            add("_Not available._")
            add("")

        # --- Spectral fidelity ---
        if self.spectral_fidelity and (self.spectral_fidelity.get("bands") or []):
            add("## Spectral fidelity (band means)")
            add("")
            add("| Band | 10 m mean | 2.5 m mean | Delta | Relative |")
            add("| --- | ---: | ---: | ---: | ---: |")
            for entry in self.spectral_fidelity["bands"]:
                add(
                    f"| {entry.get('band')} | {_num(entry.get('lr_mean'), 4)} "
                    f"| {_num(entry.get('sr_mean'), 4)} | {_num(entry.get('delta'), 5)} "
                    f"| {_num(entry.get('rel_pct'), 2)} % |"
                )
            add("")

        # --- Thematic products ---
        if self.indices:
            add("## Thematic products")
            add("")
            add("| Index | Mean | P05 | P95 | Valid | Edge gain |")
            add("| --- | ---: | ---: | ---: | ---: | ---: |")
            for stat in self.indices:
                delta = stat.get("delta") or {}
                valid = stat.get("valid_fraction")
                valid_pct = _num(valid * 100.0, 1) if isinstance(valid, (int, float)) else "—"
                add(
                    f"| {stat.get('name', stat.get('key'))} | {_num(stat.get('mean'), 3)} "
                    f"| {_num(stat.get('p05'), 3)} | {_num(stat.get('p95'), 3)} "
                    f"| {valid_pct} % | {_num(delta.get('edge_gain'), 2)} |"
                )
            add("")
            add(
                "_Edge gain compares the thematic gradient energy of the 2.5 m index against the "
                "interpolated 10 m index; above 1.0 means the product resolves boundaries the "
                "native index could not._"
            )
            add("")

        # --- Caveats ---
        add("## Caveats")
        add("")
        for caveat in self.summary.get("caveats", []):
            add(f"> {caveat}")
            add(">")
        add("")

        return "\n".join(lines)


def build_report(
    job_id: str,
    scene: dict[str, Any] | None = None,
    wald: Any = None,
    consistency: Any = None,
    uncertainty: Any = None,
    spectral_fidelity: Any = None,
    indices: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> QualityReport:
    """Assemble the quality report and derive its verdict and caveats.

    The verdict rule is deliberately conservative:

    * ``validated`` — the consistency check passed, the Wald SSIM is at least 0.80, and the
      reliability score is at least 70.
    * ``inconclusive`` — either nothing quantitative was measured (no Wald result and no
      consistency result), or the measurements that were taken do not support the product:
      the consistency check failed *and* the Wald SSIM is below 0.60. Reporting such a
      product as merely caveated would overstate the evidence.
    * ``validated-with-caveats`` — everything in between.

    Parameters
    ----------
    job_id : str
        Job identifier.
    scene : dict, optional
        Free-form job metadata to embed.
    wald, consistency, uncertainty : dataclass or dict, optional
        Assessment results; either the dataclass or its ``to_dict()`` output is accepted.
    spectral_fidelity : dict, optional
        Output of :func:`ntro_srm.evaluation.consistency.spectral_fidelity`.
    indices : list[dict], optional
        Per-index statistics.
    generated_at : str, optional
        ISO-8601 timestamp; defaults to now in UTC.

    Returns
    -------
    QualityReport
        The assembled report.
    """
    wald_d = _as_dict(wald)
    cons_d = _as_dict(consistency)
    unc_d = _as_dict(uncertainty)
    fidelity_d = _as_dict(spectral_fidelity)

    wald_metrics = (wald_d or {}).get("metrics") or {}
    ssim = wald_metrics.get("ssim")
    reliability = (unc_d or {}).get("reliability_score")
    consistency_passed = bool((cons_d or {}).get("passed"))

    def _at_least(value: Any, threshold: float) -> bool:
        """True when a numeric value is present, finite and meets the threshold."""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return False
        return f == f and f >= threshold

    if wald_d is None and cons_d is None:
        verdict = "inconclusive"
    elif consistency_passed and _at_least(ssim, 0.80) and _at_least(reliability, 70.0):
        verdict = "validated"
    elif cons_d is not None and not consistency_passed and not _at_least(ssim, 0.60):
        # Measured, and the measurements argue against the product.
        verdict = "inconclusive"
    else:
        verdict = "validated-with-caveats"

    headline: list[dict[str, Any]] = []
    for key in HEADLINE_KEYS:
        if key not in wald_metrics:
            continue
        meta = _metrics.METRIC_META.get(key, {})
        headline.append(
            {
                "key": key,
                "label": meta.get("label", key),
                "value": wald_metrics.get(key),
                "unit": meta.get("unit", ""),
                "better": meta.get("better", "higher"),
                "rating": _rate(wald_metrics.get(key), meta),
            }
        )
    if reliability is not None:
        headline.append(
            {
                "key": "reliability_score",
                "label": "Reliability",
                "value": reliability,
                "unit": "%",
                "better": "higher",
                "rating": _rate(reliability, {"better": "higher", "good": 70.0, "excellent": 85.0}),
            }
        )
    if cons_d is not None:
        headline.append(
            {
                "key": "consistency",
                "label": "Radiometric consistency",
                "value": 1.0 if consistency_passed else 0.0,
                "unit": "",
                "better": "higher",
                "rating": "excellent" if consistency_passed else "fair",
            }
        )

    caveats: list[str] = [RECONSTRUCTION_CAVEAT]
    if wald_d is not None:
        caveats.append(SCALE_CAVEAT)
    else:
        caveats.append(
            "Wald synthesis validation was not run for this product, so no quantitative "
            "accuracy figure is available."
        )
    if cons_d is None:
        caveats.append("The radiometric consistency check did not run for this product.")
    elif not consistency_passed:
        caveats.append(
            "The super-resolved product exceeded the radiometric consistency tolerance when "
            "downsampled back to 10 m. Treat its reflectance values as indicative only."
        )
    if unc_d is None:
        caveats.append("No uncertainty estimate accompanies this product.")
    elif int(unc_d.get("n_ensemble", 1) or 1) <= 1:
        caveats.append(
            "Uncertainty was estimated without a test-time-augmentation ensemble, so the "
            "confidence field reflects synthesised-detail magnitude rather than model spread."
        )
    if not indices:
        caveats.append("No thematic index products were derived for this run.")

    return QualityReport(
        job_id=job_id,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        scene=scene or {},
        wald=wald_d,
        consistency=cons_d,
        uncertainty=unc_d,
        spectral_fidelity=fidelity_d,
        indices=indices,
        summary={"verdict": verdict, "headline_metrics": headline, "caveats": caveats},
    )


def save_report(report: QualityReport, out_dir: Any) -> dict[str, str]:
    """Write the report to disk as both JSON and markdown.

    Parameters
    ----------
    report : QualityReport
        Report to persist.
    out_dir : str or Path
        Destination directory, created if missing.

    Returns
    -------
    dict[str, str]
        ``{"json": <path>, "markdown": <path>}``.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "quality_report.json"
    md_path = directory / "quality_report.md"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}
