"""Execute preassigned comparison tasks without sampling or statistical assembly."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from threadpoolctl import threadpool_limits

from tabgen_eval._typing import IntArray
from tabgen_eval.execution.classifier.protocol import source_folds, uses_classifier_protocol
from tabgen_eval.metrics.base import Metric
from tabgen_eval.metrics.c2st import ClassifierMetric
from tabgen_eval.schema import Schema

from .details import ConcentrationDetails, FidelityDetails
from .dispatch import compute_python_details, supports_details

if TYPE_CHECKING:
    from .context import ExecutionContext, NumericalContext
    from .outcomes import ClassifierOutcome


@dataclass(frozen=True)
class ComparisonTask:
    """Caller-assigned execution metadata for a pair of source-position arrays.

    The caller chooses the private random stream. Fold assignment consumes that
    stream once; neither scheduling nor result collection changes its seed.
    """

    reference: IntArray
    candidate: IntArray
    rng: np.random.Generator
    reference_folds: IntArray | None = None
    candidate_folds: IntArray | None = None
    fit_seed: int | None = None


@dataclass
class ComparisonOutcome:
    """Scalar values, optional fit diagnostics and descriptive numerical tables."""

    values: dict[str, float]
    diagnostics: dict[str, object] | None = None
    details: FidelityDetails | ConcentrationDetails | None = None


def minimum_rows(metric: Metric) -> int:
    """Combine an explicit metric minimum with intrinsic classifier fold requirements."""
    return max(metric.n_min, metric.cv_folds if uses_classifier_protocol(metric) else 0)


def comparison_task(
    metric: Metric, reference: IntArray, candidate: IntArray, *, rng: np.random.Generator
) -> ComparisonTask:
    """Prepare source-stable classifier folds or retain the sample metric's RNG."""
    if uses_classifier_protocol(metric) and min(len(reference), len(candidate)) >= minimum_rows(
        metric
    ):
        rf, sf = source_folds(len(reference), len(candidate), metric.cv_folds, rng)
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        return ComparisonTask(reference, candidate, rng, rf, sf, seed)
    return ComparisonTask(reference, candidate, rng)


def _fit_outcome(
    metric: Metric, fitted: ClassifierOutcome, task: ComparisonTask
) -> ComparisonOutcome:
    return ComparisonOutcome(
        {metric.stat_names[0]: float(fitted.value)},
        dict(
            raw_value=float(fitted.value),
            converged=bool(fitted.converged),
            folds=fitted.folds,
            fit_seed=task.fit_seed,
            point=True,
        ),
    )


def _unavailable(metric: Metric, task: ComparisonTask) -> ComparisonOutcome:
    classifier = uses_classifier_protocol(metric)
    reason = (
        "fewer sampled rows per class than classifier folds"
        if classifier and min(len(task.reference), len(task.candidate)) < metric.cv_folds
        else "fewer sampled rows than the metric minimum"
    )
    diagnostics: dict[str, object] = dict(reason=reason, n_min=minimum_rows(metric))
    if classifier:
        diagnostics.update(
            raw_value=float("nan"), converged=False, folds=[], fit_seed=None, point=True
        )
    return ComparisonOutcome(dict.fromkeys(metric.stat_names, float("nan")), diagnostics)


def _python_task(
    metric: Metric,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    task: ComparisonTask,
    *,
    fit_threads: int,
    details: bool,
) -> ComparisonOutcome:
    real = reference.iloc[task.reference].reset_index(drop=True)
    fake = candidate.iloc[task.candidate].reset_index(drop=True)
    if uses_classifier_protocol(metric):
        budget = (
            {"fit_threads": fit_threads}
            if type(metric).compute_partitioned is ClassifierMetric.compute_partitioned
            else {}
        )
        with threadpool_limits(fit_threads):
            fitted = metric.compute_partitioned(
                real,
                fake,
                schema,
                cast("IntArray", task.reference_folds),
                cast("IntArray", task.candidate_folds),
                seed=cast("int", task.fit_seed),
                point=True,
                **budget,
            )
        return _fit_outcome(metric, fitted, task)
    if details and supports_details(metric):
        result = compute_python_details(metric, real, fake, schema)
        return ComparisonOutcome(result.values, details=result.details)
    return ComparisonOutcome(metric.compute(real, fake, schema, rng=task.rng))


def _row_matrix(rows: Sequence[IntArray]) -> IntArray:
    """Keep a single task's potentially large index vector as a read-only-use view."""
    return rows[0][None, :] if len(rows) == 1 else np.stack(rows)


def execute_comparisons(
    metric: Metric,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    tasks: Iterable[ComparisonTask],
    *,
    context: NumericalContext | None = None,
    ideal: bool = False,
    n_jobs: int = 1,
    parallel: bool = False,
    details: bool = False,
    limit_inner_threads: bool = True,
) -> list[ComparisonOutcome]:
    """Execute ordered tasks through one backend, retaining finite nonconverged fits.

    Inputs are validated eligible source pools. ``ideal`` addresses both
    numerical roles through the context's reference storage. Point calls run
    sequentially with the full CPU thread budget; repeated Python calls divide
    that budget between workers and classifier fits. Raw outcomes are returned
    without interval estimation, labels or report construction.
    """
    tasks = list(tasks)
    if not tasks:
        return []
    result: list[ComparisonOutcome | None] = [None] * len(tasks)
    eligible: list[tuple[int, ComparisonTask]] = []
    minimum = minimum_rows(metric)
    for index, task in enumerate(tasks):
        if min(len(task.reference), len(task.candidate)) < minimum:
            result[index] = _unavailable(metric, task)
        else:
            eligible.append((index, task))
    if not eligible:
        return cast("list[ComparisonOutcome]", result)
    live = [task for _, task in eligible]
    classifier = uses_classifier_protocol(metric)
    if context is not None:
        rd = _row_matrix([task.reference for task in live])
        sd = _row_matrix([task.candidate for task in live])
        if classifier:
            assert isinstance(metric, ClassifierMetric)
            fitted = context.compute_classifier(
                metric,
                rd,
                sd,
                _row_matrix([cast("IntArray", task.reference_folds) for task in live]),
                _row_matrix([cast("IntArray", task.candidate_folds) for task in live]),
                [cast("int", task.fit_seed) for task in live],
                ideal=ideal,
                point=True,
                fit_threads=(
                    min(metric.bootstrap_fit_threads, context.workers)
                    if parallel
                    else context.workers
                ),
            )
            computed = [
                _fit_outcome(metric, fit, task) for fit, task in zip(fitted, live, strict=True)
            ]
        elif details:
            computed = []
            for task in live:
                value = cast("ExecutionContext", context).compute_detailed(
                    metric, task.reference, task.candidate, ideal=ideal
                )
                computed.append(ComparisonOutcome(value.values, details=value.details))
        else:
            values = cast("ExecutionContext", context).compute(metric, rd, sd, ideal=ideal)
            computed = [
                ComparisonOutcome({name: float(scores[i]) for name, scores in values.items()})
                for i in range(len(live))
            ]
    else:
        total = effective_n_jobs(n_jobs)
        threads = min(getattr(metric, "bootstrap_fit_threads", 1), total) if parallel else total
        workers = max(1, total // threads) if parallel else 1
        if getattr(metric, "execution_backend", None) == "xgboost" and metric.device.startswith(
            "cuda"
        ):
            workers = 1
        if classifier or workers > 1:
            from tabgen_eval.execution.runtime.memory import (
                classifier_minimum_bytes,
                plan_metric_memory,
                process_worker_bytes,
            )

            rows = max(len(task.reference) + len(task.candidate) for task in live)
            minimum = (
                classifier_minimum_bytes(
                    metric,
                    rows,
                    len(schema.numeric),
                    len(schema.categorical),
                    tuple(int(reference[column].nunique()) for column in schema.categorical),
                )
                if classifier
                else rows * max(1, len(schema.numeric) + len(schema.categorical)) * 64 + 1024**2
            )
            plan = plan_metric_memory(
                metric,
                "cpu",
                concurrent_tasks=min(workers, len(live)),
                minimum_task_bytes=minimum,
                worker_overhead_bytes=process_worker_bytes() if workers > 1 else 0,
            )
            workers = plan.concurrent_tasks
        calls = [
            delayed(_python_task)(
                metric, reference, candidate, schema, task, fit_threads=threads, details=details
            )
            for task in live
        ]
        if workers == 1:
            computed = [function(*args, **kwargs) for function, args, kwargs in calls]
        elif classifier or limit_inner_threads:
            with parallel_config(backend="loky", inner_max_num_threads=threads):
                computed = Parallel(n_jobs=workers)(calls)
        else:
            computed = Parallel(n_jobs=workers)(calls)
    for (index, _), value in zip(eligible, computed, strict=True):
        result[index] = value
    return cast("list[ComparisonOutcome]", result)


def comparison_batch_size(
    metric: Metric,
    n: int,
    schema: Schema,
    *,
    context: NumericalContext | None = None,
    pools: int = 2,
) -> int:
    """Bound native task batches or Python index/frame work with an explicit role count."""
    if context is not None:
        return context.batch_size(metric, n)
    width = len(schema.numeric) + len(schema.categorical)
    return max(
        1,
        min(64, (8 * 1024**2) // (8 * pools * n), (256 * 1024**2) // max(8 * pools * n * width, 1)),
    )
