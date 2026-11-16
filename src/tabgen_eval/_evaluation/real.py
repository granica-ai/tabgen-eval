"""Evaluate disjoint test-only comparisons using the shared source planner."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum

import numpy as np
import pandas as pd

from tabgen_eval._options import ResolvedOptions, describe_options
from tabgen_eval.execution.runtime.domains import metric_input_schema
from tabgen_eval.execution.runtime.preparation import prepare_metric_sessions
from tabgen_eval.execution.runtime.sampling import SamplingPlan
from tabgen_eval.metrics.base import Metric
from tabgen_eval.report import Report
from tabgen_eval.report.evaluation import reference_report
from tabgen_eval.schema import Schema, validate
from tabgen_eval.schema.validate import _require_dataframe
from tabgen_eval.statistics.constants import DEFAULT_CONFIDENCE_INTERVAL
from tabgen_eval.uncertainty.reference import compute_distribution
from tabgen_eval.uncertainty.results import (
    RealMetricScores,
    RealReferenceMetadata,
    RealReferenceResult,
    RealSampling,
)


def _plain(value: object) -> object:
    """Own JSON-compatible option metadata without converting category values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_plain(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _metric_options(metric: Metric) -> dict[str, object]:
    """Record resolved numerical controls, including inherited class defaults."""
    options: dict[str, object] = {}
    for cls in reversed(type(metric).__mro__):
        options.update(
            {
                name: _plain(value)
                for name, value in vars(cls).items()
                if not name.startswith("_")
                and isinstance(value, (str, int, float, bool, tuple, list, dict, type(None)))
            }
        )
    options.update(
        {name: _plain(value) for name, value in vars(metric).items() if not name.startswith("_")}
    )
    return options


def compute_real_reference(
    real: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[Metric],
    options: ResolvedOptions,
) -> RealReferenceResult:
    """Collect raw real-only values at half of each metric's eligible source size.

    Fixed evaluation executes one actual draw. Uncertainty requests the resolved
    number of fresh disjoint comparisons; rows can recur across repetitions.
    MLE fits and scores real models only and returns its raw ``score`` vector.
    No generated-trained score, ratio, or placeholder generated model is made.
    All exclusion policies run before draws, while fitted transforms remain
    local to their training observations or comparison geometry.
    """
    _require_dataframe(real, "real")
    if not isinstance(schema, Schema):
        raise TypeError("schema must be an explicit Schema instance")
    if not metrics or any(metric.category == "memorization" for metric in metrics):
        raise ValueError("Test-only evaluation requires resolved comparison metrics")
    validate(real, real, metric_input_schema(schema, metrics))
    for metric in metrics:
        metric.validate_inputs(real, real, schema)
    count = 1 if options["uncertainty"] == "none" else options["n_replicates"]
    domains = prepare_metric_sessions(
        metrics, {"test": real}, schema, options=options, n_replicates=count
    )
    results: dict[str, RealMetricScores] = {}
    sampling: dict[str, RealSampling] = {}
    for prepared in domains:
        domain, plan = prepared.source, prepared.sampling
        assert isinstance(plan, SamplingPlan)
        source = domain.pools["test"]
        for metric in prepared.metrics:
            scores, execution, diagnostics = compute_distribution(
                metric,
                source,
                schema,
                plan,
                session=prepared.session,
                n_replicates=count,
                seed=options["seed"],
                n_jobs=options["n_jobs"],
            )
            sampling[metric.name] = {
                "n_source": plan.n_source,
                "n_res": plan.n_res,
                "policy": domain.policy or "all_rows",
                "replacement": False,
                "real_roles_disjoint": True,
            }
            results[metric.name] = RealMetricScores(
                metric.name,
                scores,
                execution,
                _metric_options(metric),
                diagnostics,
                sampling[metric.name],
                domain.row_usage(plan.n_res),
            )
    results = {metric.name: results[metric.name] for metric in metrics}
    resolutions = {item["n_res"] for item in sampling.values()}
    n_res = next(iter(resolutions)) if len(resolutions) == 1 else None
    metadata = RealReferenceMetadata(
        n_source=len(real),
        n_res=n_res,
        n_replicates=count,
        seed=deepcopy(options["seed"]),
        confidence_interval=DEFAULT_CONFIDENCE_INTERVAL,
        schema=schema.to_dict(),
        schema_inferred=False,
        per_metric_sampling=sampling,
    )
    return RealReferenceResult(results, metadata)


def evaluate_test_only(
    real: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[Metric],
    options: ResolvedOptions,
) -> Report:
    """Build a scalar or uncertainty report from the same real-only raw outcomes."""
    reference = compute_real_reference(real, schema, metrics=metrics, options=options)
    return reference_report(
        reference,
        options_dict=describe_options(options),
        point=options["uncertainty"] == "none",
        metrics=metrics,
    )
