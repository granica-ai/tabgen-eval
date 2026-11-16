"""Select real-only repeated draws and collect raw comparison outcomes."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from threadpoolctl import threadpool_limits

from tabgen_eval._typing import FloatArray, IntArray, SeedLike
from tabgen_eval.execution.classifier.protocol import uses_classifier_protocol
from tabgen_eval.execution.runtime.comparisons import (
    comparison_batch_size,
    comparison_task,
    execute_comparisons,
)
from tabgen_eval.execution.runtime.metadata import execution_info
from tabgen_eval.execution.runtime.sampling import SamplingPlan, draw_batches

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.comparisons import ComparisonTask
    from tabgen_eval.execution.runtime.context import ContextProvider
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.metrics.utility import MLE
    from tabgen_eval.schema import Schema


def classifier_task(
    metric: Metric,
    reference: IntArray,
    candidate: IntArray,
    *,
    seed: SeedLike,
    replicate: int,
) -> ComparisonTask:
    """Prepare a real-only classifier task using its independent numerical stream."""
    metric_id = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    sequence = np.random.SeedSequence(seed, spawn_key=(metric_id, replicate, 0xC2A1))
    return comparison_task(metric, reference, candidate, rng=np.random.default_rng(sequence))


def compute_distribution(
    metric: Metric,
    real: pd.DataFrame,
    schema: Schema,
    plan: SamplingPlan,
    *,
    session: ContextProvider,
    n_replicates: int,
    seed: SeedLike,
    n_jobs: int,
) -> tuple[dict[str, FloatArray], dict[str, object], tuple[dict[str, object], ...]]:
    """Collect ordered real-real vectors with common execution and minimum checks."""
    if plan.n_res < max(1, metric.n_min):
        return (
            {name: np.full(n_replicates, np.nan) for name in metric.stat_names},
            {"implementation": "unavailable", "device": "cpu"},
            tuple(
                {
                    "replicate": index,
                    "reason": "fewer eligible sampled rows than the metric minimum",
                    "n_min": max(1, metric.n_min),
                    "insufficient_data": True,
                }
                for index in range(n_replicates)
            ),
        )
    if metric.name == "mle":
        return _mle_distribution(
            cast("MLE", metric),
            real,
            schema,
            plan,
            n_replicates=n_replicates,
            seed=seed,
            n_jobs=n_jobs,
        )
    context = session.context_for(metric)
    classifier = uses_classifier_protocol(metric)
    values = {name: np.full(n_replicates, np.nan) for name in metric.stat_names}
    diagnostics: list[dict[str, object]] = []
    size = comparison_batch_size(metric, plan.n_res, schema, context=context)
    for batch in draw_batches(plan, n_replicates=n_replicates, seed=seed, batch_size=size):
        tasks = []
        for index, (reference, candidate) in enumerate(
            zip(batch.reference, batch.candidate, strict=True), batch.start
        ):
            if classifier:
                task = classifier_task(metric, reference, candidate, seed=seed, replicate=index)
            else:
                rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index, 0xCA1)))
                task = comparison_task(metric, reference, candidate, rng=rng)
            tasks.append(task)
        outcomes = execute_comparisons(
            metric,
            real,
            real,
            schema,
            tasks,
            context=context,
            ideal=True,
            n_jobs=n_jobs,
            parallel=True,
            limit_inner_threads=False,
        )
        selection = slice(batch.start, batch.start + len(tasks))
        for name in values:
            values[name][selection] = [outcome.values.get(name, np.nan) for outcome in outcomes]
        for index, outcome in enumerate(outcomes, batch.start):
            if outcome.diagnostics is not None:
                diagnostics.append(dict(replicate=index, **outcome.diagnostics))
    return values, execution_info(metric, context), tuple(diagnostics)


def _mle_reference_task(
    metric: MLE,
    real: pd.DataFrame,
    schema: Schema,
    reference: IntArray,
    training: IntArray,
    *,
    seed: SeedLike,
    replicate: int,
    threads: int,
) -> tuple[float, dict[str, object]]:
    """Fit one real model and score only the disjoint real scoring observations."""
    if len(reference) < max(1, metric.n_min):
        return np.nan, {
            "reason": "fewer sampled rows than the metric minimum",
            "n_min": metric.n_min,
        }
    scoring, train = real.iloc[reference], real.iloc[training]
    if "score" in metric.not_applicable(scoring, train, schema):
        return np.nan, {"reason": "no predictor columns", "not_applicable": True}
    metric_id = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(metric_id, replicate, 0xCA1))
    )
    with threadpool_limits(threads):
        outcome = metric.compute_explicit_with_budget(
            scoring,
            train,
            train,
            schema,
            rng=rng,
            fit_threads=threads,
            point=True,
            baseline_only=True,
        )
    return outcome.reference_scores["score"], outcome.diagnostics


def _mle_distribution(
    metric: MLE,
    real: pd.DataFrame,
    schema: Schema,
    plan: SamplingPlan,
    *,
    n_replicates: int,
    seed: SeedLike,
    n_jobs: int,
) -> tuple[dict[str, FloatArray], dict[str, object], tuple[dict[str, object], ...]]:
    """Collect real-trained MLE scores using the shared pairwise draw sequence."""
    scores = np.full(n_replicates, np.nan)
    diagnostics: list[dict[str, object]] = []
    threads = metric.fit_threads
    from tabgen_eval.execution.xgboost.device import select_device

    predictors = tuple(c for c in schema.categorical if c != schema.target)
    task = metric.resolve_task(real, schema)
    groups = int(real[schema.target].nunique()) if task == "classification" else 1
    device = select_device(
        metric,
        rows=2 * plan.n_res,
        numeric=len(set(schema.numeric) - {schema.target}),
        cardinalities=tuple(int(real[c].nunique()) for c in predictors),
        output_groups=groups if groups > 2 else 1,
    )
    size = comparison_batch_size(metric, plan.n_res, schema)
    from tabgen_eval.execution.runtime.memory import (
        leased_metric,
        plan_metric_memory,
        process_worker_bytes,
        xgboost_workspace_bytes,
    )

    numeric = len(set(schema.numeric) - {schema.target})
    categorical = tuple(column for column in schema.categorical if column != schema.target)
    cardinalities = tuple(int(real[column].nunique()) for column in categorical)
    classes = (
        int(real[schema.target].nunique())
        if schema.target is not None and metric.resolve_task(real, schema) == "classification"
        else 1
    )
    # One real model per task: encoded predictors, preprocessing scratch and
    # the same estimated forest storage used by the model's live fit guard.
    minimum = (
        plan.n_res * (64 * numeric + 112 * len(categorical) + 32)
        + xgboost_workspace_bytes(
            plan.n_res, numeric, cardinalities, output_groups=classes if classes > 2 else 1
        )
        + 65536
    )
    for batch in draw_batches(plan, n_replicates=n_replicates, seed=seed, batch_size=size):
        workers = (
            1
            if device.startswith("cuda")
            else min(len(batch.reference), max(1, effective_n_jobs(n_jobs) // threads))
        )
        memory = plan_metric_memory(
            metric,
            device,
            concurrent_tasks=workers,
            minimum_task_bytes=minimum,
            worker_overhead_bytes=process_worker_bytes() if workers > 1 else 0,
        )
        workers = memory.concurrent_tasks
        worker_metric = leased_metric(metric, memory.task_workspace_bytes)
        worker_metric.device = device
        calls = [
            delayed(_mle_reference_task)(
                worker_metric,
                real,
                schema,
                reference,
                training,
                seed=seed,
                replicate=index,
                threads=threads,
            )
            for index, (reference, training) in enumerate(
                zip(batch.reference, batch.candidate, strict=True), batch.start
            )
        ]
        if workers == 1:
            outcomes = [function(*args, **kwargs) for function, args, kwargs in calls]
        else:
            with parallel_config(backend="loky", inner_max_num_threads=threads):
                outcomes = Parallel(n_jobs=workers)(calls)
        for index, (score, detail) in enumerate(outcomes, batch.start):
            scores[index] = score
            diagnostics.append(dict(replicate=index, **detail))
    return {"score": scores}, execution_info(metric), tuple(diagnostics)
