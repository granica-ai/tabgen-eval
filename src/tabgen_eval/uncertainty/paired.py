"""Select paired repeated draws and collect raw comparison outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabgen_eval._options import ResolvedOptions
from tabgen_eval._typing import FloatArray, IntArray
from tabgen_eval.execution.runtime.comparisons import (
    comparison_batch_size,
    comparison_task,
    execute_comparisons,
)
from tabgen_eval.execution.runtime.details import ConcentrationDetails, FidelityDetails
from tabgen_eval.execution.runtime.paired import comparison_rng
from tabgen_eval.execution.runtime.sampling import FixedDraw, paired_draw_batches

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.context import ContextProvider
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.schema import Schema


def compute_replicates(
    metric: Metric,
    holdout: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    draw: FixedDraw,
    *,
    session: ContextProvider,
    options: ResolvedOptions,
    details: bool = False,
) -> tuple[
    dict[str, FloatArray],
    dict[str, FloatArray],
    dict[str, list[dict[str, object]]],
    FidelityDetails | ConcentrationDetails | None,
]:
    """Collect repeated generated/reference vectors without statistical summaries."""
    count = options["n_replicates"]
    include_candidate = options["uncertainty"] in {"candidate", "both"}
    include_reference = options["uncertainty"] in {"reference", "both"}
    generated_values = (
        {name: np.full(count, np.nan) for name in metric.stat_names} if include_candidate else {}
    )
    reference_values = (
        {name: np.full(count, np.nan) for name in metric.stat_names} if include_reference else {}
    )
    diagnostics: dict[str, list[dict[str, object]]] = {"generated": [], "real_reference": []}
    detail_tables = None
    context = session.context_for(metric)
    size = comparison_batch_size(metric, draw.n_res, schema, context=context, pools=3)
    for batch in paired_draw_batches(
        draw,
        n_replicates=count,
        seed=options["seed"],
        batch_size=size,
        include_generated=include_candidate,
    ):
        selection = slice(batch.start, batch.start + len(batch.reference))
        sides: list[tuple[bool, IntArray, dict[str, FloatArray]]] = []
        if include_reference:
            sides.append((True, batch.real_candidate, reference_values))
        if include_candidate:
            assert batch.generated is not None
            sides.insert(0, (False, batch.generated, generated_values))
        for ideal, candidates, destination in sides:
            tasks = [
                comparison_task(
                    metric,
                    reference,
                    candidate,
                    rng=comparison_rng(
                        metric, seed=options["seed"], ideal=ideal, replicate=batch.start + i
                    ),
                )
                for i, (reference, candidate) in enumerate(
                    zip(batch.reference, candidates, strict=True)
                )
            ]
            first_detailed = details and not ideal and batch.start == 0
            outcomes = execute_comparisons(
                metric,
                holdout,
                holdout if ideal else generated,
                schema,
                tasks[:1] if first_detailed else tasks,
                context=context,
                ideal=ideal,
                n_jobs=options["n_jobs"],
                parallel=True,
                details=first_detailed,
            )
            if first_detailed:
                detail_tables = outcomes[0].details
                if detail_tables is not None:
                    detail_tables.metadata.update(
                        scope="generated_vs_reference_replicate",
                        replicate=0,
                        n_reference=draw.n_res,
                        n_candidate=draw.n_res,
                        aggregate_values=dict(outcomes[0].values),
                    )
                outcomes.extend(
                    execute_comparisons(
                        metric,
                        holdout,
                        generated,
                        schema,
                        tasks[1:],
                        context=context,
                        ideal=False,
                        n_jobs=options["n_jobs"],
                        parallel=True,
                    )
                )
            for name in destination:
                destination[name][selection] = [
                    outcome.values.get(name, np.nan) for outcome in outcomes
                ]
            key = "real_reference" if ideal else "generated"
            for index, outcome in enumerate(outcomes, batch.start):
                if outcome.diagnostics is not None:
                    diagnostics[key].append(dict(replicate=index, **outcome.diagnostics))
    return generated_values, reference_values, diagnostics, detail_tables
