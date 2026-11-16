"""Adapt internal computation results to the public evaluation report contract.

Selection and report assembly do not compute scores. Shared numerical jobs keep
their own names and statistic keys; public metrics expose selected portions of
those jobs without copying scalar results or losing execution provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..execution.runtime.details import DetailTable, FidelityDetails
from ..statistics.constants import DEFAULT_CONFIDENCE_INTERVAL
from ..statistics.distributions import ScoreDistribution
from .assembly import attach_distribution, point_stat_result
from .diagnostics import DiagnosticFlag
from .io import Report
from .result import MetricResult

if TYPE_CHECKING:
    from .._options import EvaluationOptions
    from ..execution.runtime.details import ConcentrationDetails
    from ..metrics.base import Metric
    from ..registry.selection import MetricSelection
    from ..uncertainty.results import RealReferenceResult


def project_results(
    results: Sequence[MetricResult], selection: Sequence[MetricSelection]
) -> list[MetricResult]:
    """Select public metrics from completed internal computation results.

    Each selection declares ``name``, ``group``, ``computation``, ``outputs``
    and ``primary``. Some procedures intentionally produce a subset of their
    usual outputs: test-only utility contains its real-trained score alone.
    """
    sources = {result.name: result for result in results}
    projected: list[MetricResult] = []
    for spec in selection:
        source = sources.get(spec.computation)
        if source is None:
            raise ValueError(f"Missing computation result for {spec.computation!r}")
        stats = {name: source.stats[name] for name in spec.outputs if name in source.stats}
        if spec.primary not in stats:
            raise ValueError(
                f"Computation {spec.computation!r} did not produce primary statistic "
                f"{spec.primary!r} for metric {spec.name!r}"
            )
        projected.append(
            MetricResult(
                name=spec.name,
                category=spec.group,
                stats=stats,
                diagnostics=source.diagnostics,
                details=_selected_details(source.details, spec),
                primary=spec.primary,
                computation=spec.computation,
            )
        )
    return projected


def _selected_details(
    details: FidelityDetails | ConcentrationDetails | None, spec: MetricSelection
) -> FidelityDetails | ConcentrationDetails | None:
    """Retain a conditional component's descriptive scores and shared slice facts."""
    if details is None or spec.computation != "conditional_fidelity":
        return details
    if not isinstance(details, FidelityDetails):
        return details
    component = {"cond_shape": "shape", "cond_trend": "trend", "recovery": "recovery"}.get(
        spec.name
    )
    if component is None:
        return details
    all_stats = {
        "cond_shape_all",
        "cond_shape_rare",
        "cond_trend_all",
        "cond_trend_rare",
        "recovery_all",
        "recovery_rare",
    }
    tables = {}
    for name, table in details.tables.items():
        excluded: set[str] = (
            all_stats.difference(spec.outputs)
            if name == "columns"
            else {"shape", "trend", "recovery"}.difference({component})
            if name == "slices"
            else set()
        )
        tables[name] = DetailTable(
            {key: value for key, value in table.columns.items() if key not in excluded},
            metadata=table.metadata,
        )
    return replace(details, tables=tables)


def reference_report(
    reference: RealReferenceResult,
    *,
    options_dict: EvaluationOptions,
    point: bool,
    metrics: Sequence[Metric],
) -> Report:
    """Present real-only results in the same report used by paired evaluation.

    Repeated disjoint test draws provide reference distributions and their
    mean, without a fabricated generated score. With ``point=True``, the
    single actual draw is retained as a point and has no uncertainty band.
    ``metrics`` supplies resolved internal computation instances, including
    task-specific statistic display metadata. Point evaluation consumes raw
    arrays directly; it never constructs and subsequently discards a score
    distribution.
    """
    metadata = reference.metadata
    templates = {metric.name: metric for metric in metrics}
    results: list[MetricResult] = []
    for name, evaluated in reference.metrics.items():
        sampling = evaluated.sampling
        n_source = sampling["n_source"]
        n_res = sampling["n_res"]
        metric = templates[name]
        specs = {spec.name: spec for spec in metric.stats}
        result = MetricResult(
            name,
            metric.category,
            diagnostics={"replicates": list(evaluated.diagnostics), "sampling": sampling},
        )
        if evaluated.row_usage is not None:
            result.diagnostics["row_usage"] = evaluated.row_usage
        for stat_name, values in evaluated.values.items():
            if point and len(values) != 1:
                raise ValueError("A test-only point report requires exactly one draw per statistic")
            value = float(values[0]) if point else np.nan
            flags = [DiagnosticFlag.NO_CI_REQUESTED] if point else []
            if any(item.get("insufficient_data") for item in evaluated.diagnostics):
                flags.append(DiagnosticFlag.INSUFFICIENT_DATA)
            if point and np.isnan(value):
                flags.append(DiagnosticFlag.NAN_VALUE)
            stat = point_stat_result(
                specs[stat_name],
                value,
                n_holdout=n_source,
                n_generated=0,
                n_eff=n_res,
                flags=flags,
            )
            stat.score_role = "real_trained" if name == "mle" else "test_test"
            stat.point_status = None if point else "absent"
            stat.ideal_status = "absent"
            if not point:
                distribution = ScoreDistribution.from_samples(
                    values, confidence_interval=DEFAULT_CONFIDENCE_INTERVAL
                )
                attach_distribution(stat, distribution, reference=True)
                stat.value = distribution.mean
                stat.value_estimate = "mean"
                stat.point_status = None if distribution.finite_count else "absent"
            result.stats[stat_name] = stat
        results.append(result)
    provenance: dict[str, object] = {
        **metadata,
        "evaluation_kind": "test_only",
        "n_test": metadata.get("n_source", 0),
        "n_holdout": metadata.get("n_source", 0),
        "n_generated": 0,
        "input_rows": {"test": metadata.get("n_source", 0)},
        "n_eff": metadata.get("n_res"),
        "n_replicates": 0 if point else metadata.get("n_replicates", 0),
        "metric_execution": {name: item.execution for name, item in reference.metrics.items()},
        "metric_options": {name: item.options for name, item in reference.metrics.items()},
        "metric_diagnostics": {
            name: list(item.diagnostics) for name, item in reference.metrics.items()
        },
    }
    return Report(results, metadata["schema"], options_dict, provenance)
