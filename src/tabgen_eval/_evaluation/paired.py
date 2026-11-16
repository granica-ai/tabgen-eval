"""Prepare inputs and assemble fixed or repeated matched metric evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict, cast

import pandas as pd

from tabgen_eval._options import ResolvedOptions, describe_options
from tabgen_eval.execution.runtime.dispatch import supports_details
from tabgen_eval.execution.runtime.domains import metric_input_schema
from tabgen_eval.execution.runtime.metadata import execution_info
from tabgen_eval.execution.runtime.preparation import prepare_metric_sessions
from tabgen_eval.execution.runtime.sampling import FixedDraw
from tabgen_eval.metrics.base import Metric
from tabgen_eval.metrics.c2st import ClassifierMetric
from tabgen_eval.report import Report
from tabgen_eval.report.result import MetricResult
from tabgen_eval.schema import Schema, validate
from tabgen_eval.schema.validate import _require_dataframe
from tabgen_eval.statistics.constants import DEFAULT_CONFIDENCE_INTERVAL

from .estimator import Estimator


class PairedSampling(TypedDict):
    """Effective source sizes and row eligibility for one paired computation."""

    n_holdout: int
    n_generated: int
    n_res: int
    policy: str


def evaluate_pair(
    holdout: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[Metric],
    options: ResolvedOptions,
    details: bool = False,
) -> Report:
    """Run shared generated/test and test/test computations at matched resolution.

    The public planner supplies resolved internal computation instances through
    ``metrics``. Source draws, private fitting streams, numerical execution, and
    per-statistic uncertainty assembly retain their established contracts.
    """
    if not isinstance(details, bool):
        raise ValueError("details must be a boolean")
    _require_dataframe(generated, "generated")
    if any(metric.category == "memorization" for metric in metrics):
        raise ValueError("Paired evaluation supports comparison metrics only")

    _require_dataframe(holdout, "holdout")
    if not isinstance(schema, Schema):
        raise TypeError("schema must be an explicit Schema instance")
    validate(holdout, generated, metric_input_schema(schema, metrics))
    for metric in metrics:
        metric.validate_inputs(holdout, generated, schema)
    n_holdout, n_generated = len(holdout), len(generated)
    paired = any(metric.bootstrapped for metric in metrics)
    n_replicates = options["n_replicates"] if paired and options["uncertainty"] != "none" else 0
    domains = prepare_metric_sessions(
        metrics,
        {"test": holdout, "generated": generated},
        schema,
        options=options,
        n_replicates=n_replicates,
        details=details,
    )
    estimators_by_name: dict[str, Estimator] = {}
    computed: dict[str, MetricResult] = {}
    sampling: dict[str, PairedSampling] = {}
    for prepared in domains:
        domain, draw, session = prepared.source, prepared.sampling, prepared.session
        assert isinstance(draw, FixedDraw)
        reference, candidate = domain.pools["test"], domain.pools["generated"]
        metrics_in_domain = prepared.metrics
        for metric in metrics_in_domain:
            context = session.context_for(metric) if draw.n_res else None
            estimator = Estimator(
                metric,
                options,
                context=context,
                session=session,
                draw=draw,
                row_usage=domain.row_usage(draw.n_res),
                category_mismatch=False,
                details=details and supports_details(metric),
            )
            estimators_by_name[metric.name] = estimator
            computed[metric.name] = estimator.run(reference, candidate, schema)
            sampling[metric.name] = {
                "n_holdout": len(reference),
                "n_generated": len(candidate),
                "n_res": draw.n_res,
                "policy": domain.policy or "all_rows",
            }
    estimators = [estimators_by_name[metric.name] for metric in metrics]
    results = [computed[metric.name] for metric in metrics]
    resolutions = {item["n_res"] for item in sampling.values()}
    n_eff = next(iter(resolutions)) if len(resolutions) == 1 else None

    provenance: dict[str, object] = {
        "n_holdout": n_holdout,
        "n_generated": n_generated,
        "n_res": n_eff if paired else None,
        "uncertainty": options["uncertainty"],
        "n_eff": n_eff,
        "n_replicates": n_replicates,
        "confidence_interval": DEFAULT_CONFIDENCE_INTERVAL,
        "seed": options["seed"],
        "metric_execution": {
            metric.name: execution_info(metric, estimator.context)
            for metric, estimator in zip(metrics, estimators, strict=True)
        },
        "schema_inferred": False,
        "per_metric_sampling": sampling,
    }
    for metric, result in zip(metrics, results, strict=True):
        if result.diagnostics:
            default_group = (
                "classifier_diagnostics"
                if isinstance(metric, ClassifierMetric)
                else "metric_diagnostics"
            )
            group = getattr(metric, "diagnostics_group", default_group)
            entries = cast("dict[str, dict[str, object]]", provenance.setdefault(group, {}))
            entries[result.name] = result.diagnostics
    return Report(
        results=results,
        schema_dict=schema.to_dict(),
        options_dict=describe_options(options),
        provenance=provenance,
    )
