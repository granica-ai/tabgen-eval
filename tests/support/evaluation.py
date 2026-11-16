"""Build explicit numerical-engine fixtures without public selection defaults.

Production engines receive prepared metric instances and validated options.
These helpers keep frame-based tests concise while keeping fixture selection
conveniences outside the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from tabgen_eval._options import ResolvedOptions
from tabgen_eval._typing import SeedLike
from tabgen_eval.metrics import Metric
from tabgen_eval.registry.registry import REGISTRY, resolve_metrics
from tabgen_eval.report import Report
from tabgen_eval.schema import Schema
from tabgen_eval.statistics.distributions import ScoreDistribution
from tabgen_eval.uncertainty.results import RealMetricScores, RealReferenceResult
from tests.support.options import make_options, replace_options


def fixture_metrics(
    selection: Sequence[str] | Sequence[Metric], options: ResolvedOptions
) -> list[Metric]:
    """Resolve named fixture computations or use explicitly configured instances."""
    if all(isinstance(metric, Metric) for metric in selection):
        return [metric for metric in selection if isinstance(metric, Metric)]
    names: list[str] = []
    for token in selection:
        if not isinstance(token, str):
            raise TypeError("fixture metrics must all be names or all metric instances")
        matches = [spec.name for spec in REGISTRY if spec.name == token or spec.category == token]
        names.extend(matches or [token])
    return resolve_metrics(names, options)


def paired_report(
    holdout: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[str] | Sequence[Metric] = ("shape", "trend", "prdc", "conditional_fidelity"),
    options: ResolvedOptions | None = None,
    details: bool = False,
) -> Report:
    """Evaluate an explicit pair fixture using the ordinary prepared-metric path."""
    from tabgen_eval._evaluation.paired import evaluate_pair

    resolved = make_options() if options is None else options
    return evaluate_pair(
        holdout,
        generated,
        schema,
        metrics=fixture_metrics(metrics, resolved),
        options=resolved,
        details=details,
    )


def memorization_report(
    training: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[str] | Sequence[Metric] = ("dcr",),
    options: ResolvedOptions | None = None,
    holdout: pd.DataFrame | None = None,
) -> Report:
    """Evaluate configured training-reference fixture metrics."""
    from tabgen_eval._evaluation.memorization import evaluate_memorization

    resolved = make_options() if options is None else options
    return evaluate_memorization(
        training,
        generated,
        schema,
        metrics=fixture_metrics(metrics, resolved),
        options=resolved,
        holdout=holdout,
    )


def reference_scores(
    real: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[str] | Sequence[Metric] = ("shape", "trend"),
    options: ResolvedOptions | None = None,
    n_replicates: int = 10,
    seed: SeedLike | None = None,
) -> RealReferenceResult:
    """Collect raw disjoint real-pool scores with explicit repeated-draw options."""
    from tabgen_eval._evaluation.real import compute_real_reference

    resolved = make_options() if options is None else options
    resolved = replace_options(
        resolved,
        uncertainty="both",
        n_replicates=n_replicates,
        seed=resolved["seed"] if seed is None else seed,
    )
    return compute_real_reference(
        real, schema, metrics=fixture_metrics(metrics, resolved), options=resolved
    )


def summaries(metric: RealMetricScores) -> dict[str, ScoreDistribution]:
    """Summarize a raw fixture result using the report's ordinary score contract."""
    from tabgen_eval.statistics.constants import DEFAULT_CONFIDENCE_INTERVAL

    return {
        name: ScoreDistribution.from_samples(
            values, confidence_interval=DEFAULT_CONFIDENCE_INTERVAL
        )
        for name, values in metric.values.items()
    }
