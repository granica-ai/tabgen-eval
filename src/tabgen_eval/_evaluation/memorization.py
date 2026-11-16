"""Evaluate proximity to the actual data used to train a generator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from tabgen_eval._options import ResolvedOptions, describe_options
from tabgen_eval.execution.runtime.memorization import run_memorization
from tabgen_eval.report import MemorizationReport
from tabgen_eval.report.assembly import point_metric_result
from tabgen_eval.report.result import MetricResult
from tabgen_eval.schema import Schema
from tabgen_eval.schema.validate import _require_dataframe

if TYPE_CHECKING:
    from tabgen_eval.metrics.memorization import _MemorizationMetric

_SUPPORTED = frozenset({"dcr", "near_match_rate", "replication_ratio"})


def evaluate_memorization(
    training: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    holdout: pd.DataFrame | None = None,
    metrics: Sequence[_MemorizationMetric],
    options: ResolvedOptions,
) -> MemorizationReport:
    """Compare generated records with their generator's real training data.

    DCR reports q05, q10, q25, q50, q75, q90, q95 and q99 for both
    generated-to-training and holdout-to-training distances. Generated scores
    use ``qXX`` keys and holdout scores use ``holdout_qXX`` keys. The companion
    ``median`` equals generated q50; ``ratio`` divides it by holdout q50. Both
    query passes share the selected training pool and training-fitted geometry.
    Without a holdout its percentiles and the ratio are unavailable with
    ``MISSING_HOLDOUT``; unusable holdouts and zero baselines are flagged.

    Select ``near_match_rate`` to report the fraction of retained generated rows
    whose nearest training distance is at most the fixed threshold ``1e-3``. Matching is inclusive and uses the same training-fitted geometry.
    It needs no holdout; repeated generated copies each contribute to the rate.

    Select ``replication_ratio`` to report the fraction of retained training rows
    within the inclusive threshold of any generated row (restored training
    coverage). Duplicate training rows each count; generated duplicates do not
    add coverage. It needs no holdout. Empty usable generated data gives zero
    when usable training exists; empty usable training leaves it undefined.

    Every row is selected by default. DCR's optional ``max_samples`` independently
    caps training and generated pools without replacement; ``max_holdout_samples``
    independently caps holdout rows. Numeric ranges and categorical geometry are
    fitted on selected training rows. All nearest-record searches are exact.

    An explicit schema and execution settings apply. Records missing a declared
    feature are excluded before optional caps. Uncertainty settings do not affect
    this point-only procedure. The returned report retains input, selected and
    usable row counts, numerical diagnostics and execution choices, with no
    real-real reference, comparison labels or uncertainty estimates. Caller
    frames are not modified.
    """
    _require_dataframe(training, "training")
    _require_dataframe(generated, "generated")
    if holdout is not None:
        _require_dataframe(holdout, "holdout")
    if not metrics or any(metric.name not in _SUPPORTED for metric in metrics):
        raise ValueError(
            "evaluate_memorization supports dcr, near_match_rate and replication_ratio"
        )
    result = run_memorization(training, generated, schema, metrics, options, holdout=holdout)
    metadata = result.metadata
    results: list[MetricResult] = []
    for metric in metrics:
        outcome = result.outcomes[metric.name]
        metric_result = point_metric_result(
            metric,
            outcome.values,
            n_holdout=metadata.get("n_holdout") or 0,
            n_generated=metadata["n_generated"],
            n_eff=min(metadata["n_retained_training"], metadata["n_retained_generated"]),
            diagnostics=outcome.diagnostics,
            stat_flags=outcome.stat_flags,
        )
        for name, stat in metric_result.stats.items():
            stat.score_role = (
                "train_generated"
                if metric.name == "replication_ratio"
                else "test_train"
                if name.startswith("holdout_")
                else "distance_ratio"
                if name == "ratio"
                else "generated_train"
            )
            stat.ideal_status = "absent"
        results.append(metric_result)
    return MemorizationReport(
        results=results,
        schema_dict=metadata["schema"],
        options_dict=describe_options(options),
        provenance={
            **metadata,
            "evaluation_kind": "memorization",
            "uncertainty": "none",
            "n_replicates": 0,
            "metric_execution": result.execution,
            "metric_diagnostics": {
                name: value.diagnostics for name, value in result.outcomes.items()
            },
        },
    )
