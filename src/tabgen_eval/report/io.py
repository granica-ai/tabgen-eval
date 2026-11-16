"""Present scalar metric results and optional descriptive fidelity breakdowns.

Aggregate views share the registry's statistic metadata and retain values,
intervals, real-real references and flags. Detailed tables are separate views
of the fixed comparison's component scores, without per-entry uncertainty.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._options import EvaluationOptions
from ..report.diagnostics import DiagnosticFlag
from ..schema.schema import SchemaDeclaration
from ..statistics.distributions import ScoreDistribution
from .result import MetricResult, StatResult
from .roles import primary_distribution, reference_only, score_role_label
from .rows import row_usage_frame, row_usage_markdown, row_usage_summary

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from .families import FidelityReport, MemorizationReport, UtilityReport

NAN = float("nan")
EXPORT_SCHEMA_VERSION = 2


def _legend_entries(used_codes: Iterable[str]) -> list[tuple[str, str]]:
    """``(code, description)`` for each used flag, in catalogue order — the shared legend."""
    used = set(used_codes)
    return [(f.code, f.description) for f in DiagnosticFlag if f.code in used]


@dataclass
class Report:
    """Hold metric results, input/options metadata, and evaluation provenance.

    ``results`` contains MetricResult objects; ``schema_dict`` and ``options_dict``
    record the settings used. ``provenance`` retains sample sizes, execution
    choices, and fitted-model diagnostics when available. Use ``report["metric/stat"]``
    to retrieve a StatResult, a public metric name for its primary statistic,
    or a bare statistic name when it occurs only once.

    ``to_dataframe`` returns a tidy analysis table; ``summary`` and ``to_markdown``
    return readable views. ``row_usage`` shows source exclusions before sampling.
    ``details`` returns captured column, pair or slice
    tables. ``to_json`` exports values, metadata, flags, detail tables, and optional
    retained replicates as JSON text, optionally writing a file. There is no
    Report deserialization API; JSON decoding returns ordinary dictionaries.

    ``fidelity``, ``memorization`` and ``utility`` expose focused report views.
    Views share stored result objects and retain procedure-specific evaluation
    metadata; they perform no evaluation or fitting. ``plot`` creates a Plotly figure
    directly from the captured scores or details.
    """

    results: list[MetricResult]
    schema_dict: SchemaDeclaration
    options_dict: EvaluationOptions
    provenance: dict[str, object] = field(default_factory=dict[str, object])

    # -- family views and visualization ----------------------------------

    @property
    def fidelity(self) -> FidelityReport:
        """View distributional fidelity, geometry, detection and concentration."""
        from .families import FidelityReport, family_view

        return family_view(self, FidelityReport)

    @property
    def memorization(self) -> MemorizationReport:
        """View training-referenced DCR, near-match rate and replication ratio."""
        from .families import MemorizationReport, family_view

        return family_view(self, MemorizationReport)

    @property
    def utility(self) -> UtilityReport:
        """View MLE raw AUROC/MAE, generated-to-real ratios and prediction comparisons."""
        from .families import UtilityReport, family_view

        return family_view(self, UtilityReport)

    def plot(self, kind: str = "scores", **kwargs: object) -> Figure:
        """Create a Plotly figure from stored scores or captured detail tables.

        ``kind="scores"`` compares available generated and real-reference
        scores, with retained score bands when present. Other plot kinds and
        their options are documented in ``tabgen_eval.visualization.plot_report``.
        Missing details raise an actionable error; plotting never recomputes a
        metric. The returned figure supports notebook display and HTML export.
        """
        from ..visualization import plot_report

        return plot_report(self, kind=kind, **kwargs)

    # -- lookup ----------------------------------------------------------

    def __getitem__(self, key: str) -> StatResult:
        """Return a statistic by qualified key, primary metric, or unique statistic name.

        Raise KeyError for a malformed key, missing statistic or ambiguous bare name; the latter
        lists qualified alternatives. Returned objects are the stored results,
        not copies. For example, use ``"linear_c2st/auc"`` when both classifiers ran.
        """
        parts = key.split("/")
        if len(parts) > 2 or any(not part for part in parts):
            raise KeyError(f"Invalid stat key {key!r}; use 'metric/stat' or a bare statistic name")
        matches = []
        for mr in self.results:
            if len(parts) == 2 and mr.name != parts[0]:
                continue
            stat_name = parts[-1]
            if stat_name in mr.stats:
                if len(parts) != 1:
                    return mr.stats[stat_name]
                matches.append((mr.name, mr.stats[stat_name]))
            elif len(parts) == 1 and mr.name == key and mr.primary in mr.stats:
                matches.append((mr.name, mr.stats[mr.primary]))
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            alternatives = ", ".join(
                f"{name}/{getattr(stat, 'name', key)}" for name, stat in matches
            )
            raise KeyError(f"Ambiguous stat {key!r}; use 'metric/stat': {alternatives}")
        raise KeyError(f"No stat matching {key!r}")

    def iter_stats(self) -> Iterator[tuple[MetricResult, StatResult]]:
        """Yield stored ``(MetricResult, StatResult)`` pairs in report/statistic order."""
        for mr in self.results:
            for stat in mr.stats.values():
                yield mr, stat

    def details(self, metric: str, level: str | None = None) -> pd.DataFrame:
        """Return a stored descriptive table of component scores.

        ``level`` accepts ``"column"``, ``"pair"``, ``"slice"``, or ``"radius"``. The default
        is columns for Shape/Conditional, pairs for Trend, and radii
        for Concentration. These
        are point scores, with no per-entry uncertainty or quality labels.
        Enable fidelity capture with ``evaluate(..., details=True)``.
        Concentration always retains its raw point profiles and actual radii.
        Returned tables are copies and retain exact categorical label types.
        """
        from .details import detail_frame

        for result in self.results:
            if result.name == metric:
                if result.details is None:
                    raise ValueError(
                        f"No details available for {metric!r}; fidelity tables require details=True "
                        "and a computed point comparison."
                    )
                return detail_frame(result.details, metric=metric, level=level)
        raise KeyError(f"No metric matching {metric!r}")

    # -- tidy dataframe --------------------------------------------------

    def row_usage(self) -> pd.DataFrame:
        """Return source eligibility for metrics that filter rows before sampling.

        One row per public metric and source pool records the input, retained,
        and excluded counts and the excluded percentage of the original pool.
        ``policy`` distinguishes complete cases from observed utility targets;
        ``columns`` identifies the columns checked. ``n_res`` is the number of
        retained rows selected per comparison role, rather than the number of
        finite score replicates. This metadata never requires ``details=True``.
        """
        return row_usage_frame(self.results)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one analysis row per category, metric, and statistic.

        Columns include point/ideal values, interval endpoints and bootstrap medians,
        direction/target, input and selected row counts, and comma-separated
        flag codes. Model diagnostics and full replicate arrays remain in the report
        rather than being expanded into this table. ``n_eff`` counts selected
        rows per comparison role; use ``row_usage`` for source exclusions.
        """
        rows = []
        for mr, s in self.iter_stats():
            rows.append(
                {
                    "category": mr.category,
                    "metric": mr.name,
                    "stat": s.name,
                    "score_role": s.score_role,
                    "value_estimate": s.value_estimate,
                    "ideal_estimate": s.ideal_estimate,
                    "value_status": s.point_status or _value_status(s.value),
                    "ideal_status": s.ideal_status or _value_status(s.ideal),
                    "value": s.value,
                    "ci_low": s.ci_low,
                    "ci_high": s.ci_high,
                    "bootstrap_median": s.bootstrap_median,
                    "ideal": s.ideal,
                    "ideal_ci_low": s.ideal_ci_low,
                    "ideal_ci_high": s.ideal_ci_high,
                    "ideal_bootstrap_median": s.ideal_bootstrap_median,
                    "direction": s.direction.value,
                    "target": s.target,
                    "n_holdout": s.n_holdout,
                    "n_generated": s.n_generated,
                    "n_eff": s.n_eff,
                    "n_test": self.provenance.get("input_rows", {}).get("test", s.n_holdout),
                    "n_train": self.provenance.get("input_rows", {}).get("train"),
                    "flags": ",".join(s.flag_codes),
                    **_distribution_columns(s.generated_distribution, "generated"),
                    **_distribution_columns(s.reference_distribution, "reference"),
                }
            )
        return pd.DataFrame(rows)

    # -- json ------------------------------------------------------------

    def to_json(self, path: str | None = None, *, include_replicates: bool = False) -> str:
        """Export the report as JSON text and optionally write it to path.

        Results are nested by category, metric, then statistic. Nonfinite scalar
        scores/intervals become JSON null, with point status and discard reasons
        preserving their meaning. ``include_replicates=False`` omits ordered raw
        sample vectors; setting it true includes each distribution's samples.
        Schema, resolved options, and provenance are included. This is an export
        format: json.loads returns dictionaries, not reconstructed Report objects.
        Captured fidelity tables appear in a separate ``details`` subtree with
        positional columns, feature names, and typed category vocabularies.
        """
        data = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "schema": self.schema_dict,
            "options": self.options_dict,
            "provenance": self.provenance,
            "results": {},
            "metric_metadata": {
                result.name: {
                    "group": result.category,
                    "primary": result.primary,
                    "computation": result.computation or result.name,
                }
                for result in self.results
            },
            "row_usage": {
                result.name: result.diagnostics["row_usage"]
                for result in self.results
                if "row_usage" in result.diagnostics
            },
        }
        for mr, s in self.iter_stats():
            node = data["results"].setdefault(mr.category, {}).setdefault(mr.name, {})
            entry = {
                "value": _j(s.value),
                "ci_low": _j(s.ci_low),
                "ci_high": _j(s.ci_high),
                "bootstrap_median": _j(s.bootstrap_median),
                "ideal": _j(s.ideal),
                "ideal_ci_low": _j(s.ideal_ci_low),
                "ideal_ci_high": _j(s.ideal_ci_high),
                "ideal_bootstrap_median": _j(s.ideal_bootstrap_median),
                "direction": s.direction.value,
                "target": s.target,
                "n_holdout": s.n_holdout,
                "n_generated": s.n_generated,
                "n_eff": s.n_eff,
                "flags": s.flag_codes,
                "score_role": s.score_role,
                "value_estimate": s.value_estimate,
                "ideal_estimate": s.ideal_estimate,
                "value_status": s.point_status or _value_status(s.value),
                "ideal_status": s.ideal_status or _value_status(s.ideal),
                "generated_distribution": _distribution_json(
                    s.generated_distribution, include_replicates=include_replicates
                ),
                "reference_distribution": _distribution_json(
                    s.reference_distribution, include_replicates=include_replicates
                ),
            }
            node[s.name] = entry
        captured = {
            result.name: result.details for result in self.results if result.details is not None
        }
        if captured:
            from .details import details_json

            data["details"] = {name: details_json(detail) for name, detail in captured.items()}
        text = json.dumps(_json_safe(data), indent=2, allow_nan=False)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    # -- markdown --------------------------------------------------------

    def to_markdown(self, path: str | None = None) -> str:
        """Readable scores, finite-replicate counts, source exclusions, and diagnostics."""
        test_only = self.provenance.get("evaluation_kind") == "test_only"
        if test_only or any(s.score_role != "generated_test" for _, s in self.iter_stats()):
            text = _role_aware_markdown(self.results, include_reference=not test_only)
            text = _append_row_usage(text, self.results)
            if path is not None:
                with open(path, "w") as fh:
                    fh.write(text)
            return text
        lines = [
            "| Category | Metric | Stat | Generated score | Generated mean | Score band | Real-real score | Reference mean | Reference band | Finite score replicates G / R | Flags |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        used_flags: set[str] = set()
        for mr, s in self.iter_stats():
            used_flags.update(s.flag_codes)
            lines.append(
                "| {cat} | {m} | {stat} | {val} | {mean} | {ci} | {idl} | {imean} | {ici} | {counts} | {fl} |".format(
                    cat=mr.category,
                    m=mr.name,
                    stat=s.display,
                    val=_fmt(s.value, s.fmt),
                    mean=(
                        _fmt(s.generated_distribution.mean, s.fmt)
                        if s.generated_distribution is not None
                        else "—"
                    ),
                    imean=(
                        _fmt(s.reference_distribution.mean, s.fmt)
                        if s.reference_distribution is not None
                        else "—"
                    ),
                    ci=_fmt_ci(s.ci_low, s.ci_high, s.fmt),
                    idl=_fmt(s.ideal, s.fmt),
                    ici=_fmt_ci(s.ideal_ci_low, s.ideal_ci_high, s.fmt),
                    counts=_retained_counts(s),
                    fl=", ".join(s.flag_codes),
                )
            )
        legend = _legend_entries(used_flags)
        if legend:
            lines.append("")
            lines.append("**Flags**")
            lines.append("")
            lines.extend(f"- `{code}` — {desc}" for code, desc in legend)
        text = _append_row_usage("\n".join(lines), self.results)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    # -- console summary -------------------------------------------------

    def summary(self) -> str:
        """Console table retaining undefined scores and their diagnostic explanations."""
        prov = self.provenance
        if prov.get("evaluation_kind") == "test_only":
            inputs = f"test={prov.get('n_test', prov.get('n_source'))} rows"
            computed_on = "disjoint test comparisons; " + _metric_resolutions(self.results, prov)
        elif "input_rows" in prov:
            inputs = (
                ", ".join(
                    f"{role}={size}"
                    for role, size in prov["input_rows"].items()
                    if size is not None
                )
                + " rows"
            )
            computed_on = (
                _selected_training_counts(prov)
                if "n_selected_training" in prov
                else _metric_resolutions(self.results, prov)
            )
        elif prov.get("evaluation_kind") == "utility":
            inputs = (
                f"training={prov.get('n_training')}, test={prov.get('n_holdout')}, "
                f"generated={prov.get('n_generated')} rows"
            )
            computed_on = (
                f"selected training rows={prov.get('n_eff')} per role; "
                f"full test={prov.get('n_holdout')} rows"
            )
        elif "n_selected_training" in prov:
            inputs = f"training={prov.get('n_training')}, generated={prov.get('n_generated')} rows"
            computed_on = _selected_training_counts(prov)
        else:
            inputs = f"holdout={prov.get('n_holdout')}, generated={prov.get('n_generated')} rows"
            computed_on = _metric_resolutions(self.results, prov)
        coverages = sorted(
            {
                distribution.confidence_interval
                for _, stat in self.iter_stats()
                for distribution in (stat.generated_distribution, stat.reference_distribution)
                if distribution is not None
            }
        )
        uncertainty = (
            "/".join(f"{100 * coverage:g}%" for coverage in coverages) + " score bands"
            if coverages
            else "point scores"
        )
        header = (
            f"TabGenEval report   (inputs: {inputs}; {computed_on}; "
            f"B={prov.get('n_replicates')}, {uncertainty})"
        )
        lines = [header, ""]
        used_flags: set[str] = set()

        from ..registry import CATEGORY_DISPLAY_NAMES

        for category in _ordered_categories(self.results):
            cat_stats = [(mr, s) for mr, s in self.iter_stats() if mr.category == category]
            lines.append(CATEGORY_DISPLAY_NAMES.get(category, category.upper()))
            for _, s in cat_stats:
                used_flags.update(s.flag_codes)
                lines.append("  " + _summary_line(s))
            lines.append("")

        eligibility = row_usage_summary(self.results)
        if eligibility:
            lines.extend(eligibility)
            lines.append("")

        legend = _legend_entries(used_flags)
        if legend:
            lines.append("diagnostics:")
            lines.extend(f"  {code} — {desc}" for code, desc in legend)
        return "\n".join(lines).rstrip()

    # -- replicate dump --------------------------------------------------

    def dump_replicates(self, path: str) -> None:
        """Write an export including each computed distribution's ordered samples.

        Raise ValueError if no repeated-score distribution was computed.
        """
        if not any(
            stat.generated_distribution is not None or stat.reference_distribution is not None
            for _, stat in self.iter_stats()
        ):
            raise ValueError("dump_replicates requires a computed repeated-score distribution")
        self.to_json(path, include_replicates=True)


# -- formatting helpers --------------------------------------------------


def _j(x: float) -> float | None:
    """JSON-safe float (NaN -> None)."""
    return None if x is None or not np.isfinite(x) else float(x)


def _fmt(x: float, fmt: str) -> str:
    if np.isnan(x):
        return "NaN"
    return format(x, fmt)


def _fmt_ci(lo: float, hi: float, fmt: str) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"[{format(lo, fmt)}, {format(hi, fmt)}]"


def _summary_line(s: StatResult) -> str:
    if reference_only(s):
        parts = []
        if s.point_status != "absent" and s.value_estimate != "mean":
            parts.append(f"point {_fmt(s.value, s.fmt)}")
        distribution = s.reference_distribution
        if distribution is not None:
            parts.append(f"mean {_fmt(distribution.mean, s.fmt)}")
            parts.append(_fmt_ci(distribution.band_low, distribution.band_high, s.fmt))
            parts.append(
                f"finite score replicates {distribution.finite_count}/{distribution.requested_count}"
            )
        return f"{s.display:<22} {score_role_label(s)} " + ", ".join(parts)
    generated = f"{s.value_estimate} {_fmt(s.value, s.fmt)}"
    if s.score_role != "generated_test":
        generated = f"{score_role_label(s)} {generated}"
    if s.generated_distribution is not None and s.value_estimate != "mean":
        generated += f", mean {_fmt(s.generated_distribution.mean, s.fmt)}"
    generated += f" {_fmt_ci(s.ci_low, s.ci_high, s.fmt)}"
    counts = _retained_counts(s)
    suffix = f"  finite score replicates G/R {counts}" if counts != "— / —" else ""
    if np.isnan(s.ideal) and s.reference_distribution is None:
        return f"{s.display:<22} {generated}{suffix}"
    reference = f"real-real {s.ideal_estimate} {_fmt(s.ideal, s.fmt)}"
    if s.reference_distribution is not None and s.ideal_estimate != "mean":
        reference += f", mean {_fmt(s.reference_distribution.mean, s.fmt)}"
    reference += f" {_fmt_ci(s.ideal_ci_low, s.ideal_ci_high, s.fmt)}"
    return f"{s.display:<22} {generated}   {reference}{suffix}"


def _metric_resolutions(results: list[MetricResult], provenance: Mapping[str, object]) -> str:
    resolutions = {
        result.name: next(iter(result.stats.values())).n_eff for result in results if result.stats
    }
    if len(set(resolutions.values())) > 1:
        return "selected rows per role: " + ", ".join(
            f"{name}={size}" for name, size in resolutions.items()
        )
    return f"selected rows per role={next(iter(resolutions.values()), provenance.get('n_eff'))}"


def _selected_training_counts(provenance: Mapping[str, object]) -> str:
    counts = [
        f"training={provenance.get('n_selected_training')}",
        f"generated={provenance.get('n_selected_generated')}",
    ]
    if provenance.get("n_selected_holdout") is not None:
        counts.append(f"test={provenance['n_selected_holdout']}")
    return "selected " + ", ".join(counts) + " rows"


def _role_aware_markdown(results: list[MetricResult], *, include_reference: bool) -> str:
    headers = [
        "Group",
        "Metric",
        "Stat",
        "Score role",
        "Score",
        "Mean",
        "Score band",
        "Finite score replicates",
    ]
    if include_reference:
        headers.extend(
            [
                "Reference score",
                "Reference mean",
                "Reference band",
                "Reference finite score replicates",
            ]
        )
    headers.append("Flags")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    used_flags = set()
    for result in results:
        for stat in result.stats.values():
            distribution = primary_distribution(stat)
            used_flags.update(stat.flag_codes)
            cells = [
                result.category,
                result.name,
                stat.display,
                score_role_label(stat),
                "—" if stat.point_status == "absent" else _fmt(stat.value, stat.fmt),
                _fmt(distribution.mean, stat.fmt) if distribution is not None else "—",
                _fmt_ci(distribution.band_low, distribution.band_high, stat.fmt)
                if distribution is not None
                else "—",
                f"{distribution.finite_count}/{distribution.requested_count}"
                if distribution is not None
                else "—",
            ]
            if include_reference:
                reference = None if reference_only(stat) else stat.reference_distribution
                cells.extend(
                    [
                        "—" if stat.ideal_status == "absent" else _fmt(stat.ideal, stat.fmt),
                        _fmt(reference.mean, stat.fmt) if reference is not None else "—",
                        _fmt_ci(reference.band_low, reference.band_high, stat.fmt)
                        if reference is not None
                        else "—",
                        f"{reference.finite_count}/{reference.requested_count}"
                        if reference is not None
                        else "—",
                    ]
                )
            cells.append(", ".join(stat.flag_codes))
            lines.append("| " + " | ".join(cells) + " |")
    legend = _legend_entries(used_flags)
    if legend:
        lines.extend(["", "**Flags**", ""])
        lines.extend(f"- `{code}` — {description}" for code, description in legend)
    return "\n".join(lines)


def _append_row_usage(text: str, results: list[MetricResult]) -> str:
    eligibility = row_usage_markdown(results)
    return f"{text}\n\n{eligibility}" if eligibility else text


def _ordered_categories(results: list[MetricResult]) -> list[str]:
    seen: list[str] = []
    for mr in results:
        if mr.category not in seen:
            seen.append(mr.category)
    return seen


def _value_status(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "positive_infinity"
    if np.isneginf(value):
        return "negative_infinity"
    return "finite"


def _distribution_columns(distribution: ScoreDistribution | None, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_mean": distribution.mean if distribution is not None else NAN,
        f"{prefix}_requested": distribution.requested_count if distribution is not None else 0,
        f"{prefix}_retained": distribution.finite_count if distribution is not None else 0,
        f"{prefix}_discarded": distribution.unavailable_count if distribution is not None else 0,
        f"{prefix}_discarded_reasons": (
            dict(distribution.discarded_reasons) if distribution is not None else {}
        ),
    }


def _distribution_json(
    distribution: ScoreDistribution | None, *, include_replicates: bool
) -> dict[str, object] | None:
    if distribution is None:
        return None
    data = {
        "mean": distribution.mean,
        "median": distribution.median,
        "band_low": distribution.band_low,
        "band_high": distribution.band_high,
        "confidence_interval": distribution.confidence_interval,
        "requested_count": distribution.requested_count,
        "finite_count": distribution.finite_count,
        "unavailable_count": distribution.unavailable_count,
        "discarded_reasons": dict(distribution.discarded_reasons),
        "reason": distribution.reason,
    }
    if include_replicates:
        data["samples"] = distribution.samples
    return data


def _retained_counts(stat: StatResult) -> str:
    def count(distribution: ScoreDistribution | None) -> str:
        if distribution is None:
            return "—"
        return f"{distribution.finite_count}/{distribution.requested_count}"

    return f"{count(stat.generated_distribution)} / {count(stat.reference_distribution)}"


def _json_safe(value: object) -> object:
    """Encode nested numeric metadata using strict JSON and explicit scalar nulls."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _j(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
