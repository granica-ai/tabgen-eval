"""Build scalar report records and attach score distributions with factual diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike

from tabgen_eval.report.diagnostics import DiagnosticFlag
from tabgen_eval.statistics.constants import DEFAULT_CONFIDENCE_INTERVAL
from tabgen_eval.statistics.distributions import ScoreDistribution

from .result import MetricResult, StatResult

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.details import ConcentrationDetails, FidelityDetails
    from tabgen_eval.execution.runtime.utility import UtilityExecution
    from tabgen_eval.metrics.base import Metric, StatSpec
    from tabgen_eval.metrics.utility import MLE


def point_stat_result(
    spec: StatSpec,
    value: float,
    *,
    n_holdout: int,
    n_generated: int,
    n_eff: int,
    ideal: float = np.nan,
    flags: Sequence[DiagnosticFlag] = (),
) -> StatResult:
    """Construct the common scalar record; callers supply applicable factual flags."""
    return StatResult(
        name=spec.name,
        display=spec.display,
        value=float(value),
        ci_low=np.nan,
        ci_high=np.nan,
        bootstrap_median=np.nan,
        ideal=float(ideal),
        ideal_ci_low=np.nan,
        ideal_ci_high=np.nan,
        ideal_bootstrap_median=np.nan,
        direction=spec.direction,
        target=spec.target,
        n_holdout=n_holdout,
        n_generated=n_generated,
        n_eff=n_eff,
        flags=list(flags),
        fmt=spec.fmt,
    )


def attach_distribution(
    result: StatResult, distribution: ScoreDistribution, *, reference: bool = False
) -> None:
    """Attach one empirical distribution and its factual summary diagnostics."""
    if reference:
        result.reference_distribution = distribution
        result.ideal_ci_low, result.ideal_ci_high = distribution.band_low, distribution.band_high
        result.ideal_bootstrap_median = distribution.median
    else:
        result.generated_distribution = distribution
        result.ci_low, result.ci_high = distribution.band_low, distribution.band_high
        result.bootstrap_median = distribution.median
    if DiagnosticFlag.NO_CI_REQUESTED in result.flags:
        result.flags.remove(DiagnosticFlag.NO_CI_REQUESTED)
    finite = distribution.samples[np.isfinite(distribution.samples)]
    for condition, flag in (
        (len(finite) > 0 and np.all(finite == finite[0]), DiagnosticFlag.DEGENERATE_CI),
        (distribution.unavailable_count > 0, DiagnosticFlag.NONFINITE_REPLICATES),
    ):
        if condition and flag not in result.flags:
            result.flags.append(flag)


def point_metric_result(
    metric: Metric,
    values: Mapping[str, float],
    *,
    n_holdout: int,
    n_generated: int,
    n_eff: int,
    diagnostics: Mapping[str, object] | None = None,
    details: FidelityDetails | ConcentrationDetails | None = None,
    stat_flags: Mapping[str, Sequence[str | DiagnosticFlag]] | None = None,
) -> MetricResult:
    """Preserve raw point statistics without inventing references or intervals.

    Execution diagnostics retain metric-specific reasons and row counts. Known
    per-statistic flags are included in scalar views; all supplied flags remain
    in diagnostics, including codes specific to a numerical implementation.
    """
    diagnostics = dict(diagnostics or {})
    stat_flags = stat_flags or {}
    if stat_flags:
        diagnostics["stat_flags"] = stat_flags
    result = MetricResult(metric.name, metric.category, diagnostics=diagnostics, details=details)
    for spec in metric.stats:
        value = float(values.get(spec.name, np.nan))
        flags = [DiagnosticFlag.NO_CI_REQUESTED]
        if np.isnan(value):
            flags.append(DiagnosticFlag.NAN_VALUE)
        for code in stat_flags.get(spec.name, ()):
            flag = code if isinstance(code, DiagnosticFlag) else DiagnosticFlag.by_code(code)
            if flag is not None and flag not in flags:
                flags.append(flag)
        result.stats[spec.name] = point_stat_result(
            spec,
            value,
            n_holdout=n_holdout,
            n_generated=n_generated,
            n_eff=n_eff,
            flags=flags,
        )
    return result


def assemble_stat(
    *,
    spec: StatSpec,
    value_reps: ArrayLike | None,
    ideal_reps: ArrayLike | None,
    value_point: float,
    ideal_point: float,
    base_flags: Sequence[DiagnosticFlag],
    insufficient: bool,
    not_applicable: bool,
    n_holdout: int,
    n_generated: int,
    n_eff: int,
) -> StatResult:
    """Report repetition means where requested, otherwise single scores.

    Metric-specific applicability and data requirements determine whether a
    statistic can be computed. Nonfinite draws are counted and excluded from
    descriptive summaries; finite nonconverged scores remain in distributions.
    """
    result = point_stat_result(
        spec,
        value_point,
        ideal=ideal_point if spec.compare_to_ideal else float("nan"),
        n_holdout=n_holdout,
        n_generated=n_generated,
        n_eff=n_eff,
        flags=base_flags,
    )
    if not_applicable or insufficient:
        result.flags.append(
            DiagnosticFlag.NOT_APPLICABLE if not_applicable else DiagnosticFlag.INSUFFICIENT_DATA
        )
    for values, reference_side in ((value_reps, False), (ideal_reps, True)):
        if values is None or (reference_side and not spec.compare_to_ideal):
            continue
        distribution = ScoreDistribution.from_samples(
            values, confidence_interval=DEFAULT_CONFIDENCE_INTERVAL
        )
        attach_distribution(result, distribution, reference=reference_side)
        if reference_side:
            result.ideal = distribution.mean
            result.ideal_estimate = "mean"
        else:
            result.value = distribution.mean
            result.value_estimate = "mean"
    if not np.isfinite(result.value):
        result.flags.append(DiagnosticFlag.NAN_VALUE)
    if result.generated_distribution is None and result.reference_distribution is None:
        result.flags.append(DiagnosticFlag.NO_CI_REQUESTED)
    return result


def utility_metric_result(
    metric: MLE,
    outcome: UtilityExecution,
    *,
    base_flags: Sequence[DiagnosticFlag],
    unavailable: set[str],
) -> MetricResult:
    """Assemble raw predictive scores, task-specific ratios, and agreement values.

    Reference repetitions reuse the generated model; candidate-only and both-side
    repetitions refit it. Execution owns that distinction and supplies the actual unshifted
    outcomes here. Infinite ratios retain their explicit discarded reasons.
    """
    from tabgen_eval.execution.xgboost.mle import outcome_values

    point, metadata = outcome.point, outcome.metadata
    values = outcome_values(point)
    result = MetricResult(
        metric.name,
        metric.category,
        diagnostics={
            "point": {} if outcome.replicates else asdict(point),
            "replicates": [asdict(item) for item in outcome.replicates],
            "sampling": metadata,
            "execution": outcome.execution,
        },
    )
    flags = list(base_flags)
    if any(
        fit.get("converged") is False
        for item in (point, *outcome.replicates)
        for fit in cast(
            "Mapping[str, Mapping[str, object]]", item.diagnostics.get("fits", {})
        ).values()
    ):
        flags.append(DiagnosticFlag.CLASSIFIER_NOT_CONVERGED)
    for spec in metric.stats:
        repeated = (
            [outcome_values(item)[spec.name] for item in outcome.replicates]
            if outcome.replicates and spec.name not in unavailable
            else None
        )
        references = (
            [item.reference_score for item in outcome.replicates]
            if outcome.replicates
            and spec.name == "score"
            and metadata["uncertainty"] in {"reference", "both"}
            else None
        )
        stat = assemble_stat(
            spec=spec,
            value_reps=None if spec.name == "ratio" else repeated,
            ideal_reps=references,
            value_point=values[spec.name],
            ideal_point=point.reference_score if spec.name == "score" else np.nan,
            base_flags=flags,
            insufficient=bool(point.diagnostics.get("insufficient_data", False)),
            not_applicable=spec.name in unavailable,
            n_holdout=metadata["n_holdout"],
            n_generated=metadata["n_generated"],
            n_eff=metadata["n_res"],
        )
        if spec.name == "score" and outcome.replicates and references is None:
            reference_summary = ScoreDistribution.from_samples(
                [item.reference_score for item in outcome.replicates],
                confidence_interval=DEFAULT_CONFIDENCE_INTERVAL,
            )
            stat.ideal = reference_summary.mean
            stat.ideal_estimate = "mean"
        if spec.name == "ratio":
            if repeated is None and np.isposinf(point.ratio) and point.reason is None:
                stat.flags.remove(DiagnosticFlag.NAN_VALUE)
            if repeated is not None:
                reasons = Counter(
                    item.reason or ("infinite_ratio" if np.isinf(item.ratio) else "undefined_ratio")
                    for item in outcome.replicates
                    if not np.isfinite(item.ratio)
                )
                distribution = ScoreDistribution.from_samples(
                    repeated,
                    confidence_interval=DEFAULT_CONFIDENCE_INTERVAL,
                    discarded_reasons=reasons,
                )
                attach_distribution(stat, distribution)
                stat.value = distribution.mean
                stat.value_estimate = "mean"
                if not np.isfinite(stat.value) and DiagnosticFlag.NAN_VALUE not in stat.flags:
                    stat.flags.append(DiagnosticFlag.NAN_VALUE)
                elif np.isfinite(stat.value) and DiagnosticFlag.NAN_VALUE in stat.flags:
                    stat.flags.remove(DiagnosticFlag.NAN_VALUE)
        result.stats[spec.name] = stat
    return result
