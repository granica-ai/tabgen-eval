"""Execute one fixed comparison and its paired real-real yardstick."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabgen_eval._typing import SeedLike
from tabgen_eval.execution.classifier.protocol import uses_classifier_protocol

from .comparisons import comparison_task, execute_comparisons
from .metadata import execution_info
from .outcomes import FixedMetricResult
from .sampling import FixedDraw

if TYPE_CHECKING:
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.schema import Schema

    from .context import ContextProvider


def comparison_rng(
    metric: Metric, *, seed: SeedLike, ideal: bool, replicate: int | None = None
) -> np.random.Generator:
    """Create a private metric/comparison stream independent of row sampling.

    Generated and real-real fits use distinct streams. They share exactly the
    same reference observations and ordering, while fold assignments and model
    randomness are local to each comparison.
    """
    metric_id = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    key = (0xF17ED, metric_id, int(ideal))
    if replicate is not None:
        key += (replicate + 1,)
    sequence = np.random.SeedSequence(seed, spawn_key=key)
    return np.random.default_rng(sequence)


def compute_fixed_metric(
    metric: Metric,
    holdout: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    draw: FixedDraw,
    *,
    session: ContextProvider,
    seed: SeedLike,
    n_jobs: int,
    details: bool = False,
    include_reference: bool = True,
    include_generated: bool = True,
) -> FixedMetricResult:
    """Compute the requested unrepeated sides over a shared sampled reference.

    Both comparisons reuse the same prepared source context. Numerical
    implementations own draw-local geometry and training-only transformations;
    this adapter supplies indices, source-stable folds and private fit seeds.
    The executor enforces explicit metric minima and intrinsic classifier fold
    requirements. Applicability is draw-local; classifier failures remain in
    diagnostics.
    """
    context = session.context_for(metric)
    diagnostics: dict[str, object] = {}
    detail_tables = None
    results: dict[bool, dict[str, float]] = {}
    sides = [(False, draw.generated)] if include_generated else []
    if include_reference:
        sides.append((True, draw.real_candidate))
    for ideal, candidate in sides:
        task = comparison_task(
            metric,
            draw.reference,
            candidate,
            rng=comparison_rng(metric, seed=seed, ideal=ideal),
        )
        outcome = execute_comparisons(
            metric,
            holdout,
            holdout if ideal else generated,
            schema,
            [task],
            context=context,
            ideal=ideal,
            n_jobs=n_jobs,
            details=details and not ideal,
        )[0]
        if uses_classifier_protocol(metric):
            diagnostics["real_reference" if ideal else "generated"] = outcome.diagnostics
        if outcome.details is not None:
            outcome.details.metadata.update(
                scope="generated_vs_reference_point",
                n_reference=draw.n_res,
                n_candidate=draw.n_res,
            )
            detail_tables = outcome.details
        results[ideal] = {name: float(value) for name, value in outcome.values.items()}
    return FixedMetricResult(
        metric.name,
        results.get(False, {}),
        results.get(True, {}),
        execution_info(metric, context),
        diagnostics,
        detail_tables,
    )
