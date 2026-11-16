"""Fit and score predictive utility on fixed or repeated evaluation draws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

from tabgen_eval._options import ResolvedOptions
from tabgen_eval._typing import IntArray, SeedLike, TaskKind
from tabgen_eval.execution.xgboost.prediction import fit_predictor
from tabgen_eval.execution.xgboost.utility import MLEOutcome, score_mle

from .domains import SourceDomain, eligible_fixed_draw
from .sampling import FixedDraw
from .utility_sampling import UtilityDraw, utility_draws, utility_point_draw

if TYPE_CHECKING:
    from tabgen_eval.execution.xgboost.prediction import FittedPredictor
    from tabgen_eval.metrics.utility import MLE
    from tabgen_eval.preprocessing.sources import ObservationPrecision
    from tabgen_eval.schema import Schema


class _UtilityPrecision(TypedDict, total=False):
    preprocessing: ObservationPrecision


class UtilityMetadata(_UtilityPrecision):
    """Resolved source roles and sampling policy for raw predictive scores."""

    n_holdout: int
    n_real_train: int
    n_real_test: int
    separate_real_pools: bool
    n_generated: int
    n_res: int
    n_selected_generated: int
    n_selected_real_train: int
    n_selected_real_test: int
    uncertainty: str
    n_replicates: int
    sampling: str
    replacement: bool
    real_roles_disjoint: bool
    shared_real_scoring: bool
    source: str
    generated_training: str
    task: TaskKind


@dataclass(frozen=True)
class UtilityExecution:
    """Single outcome or first repetition, repeated outcomes, and provenance."""

    point: MLEOutcome
    replicates: tuple[MLEOutcome, ...]
    metadata: UtilityMetadata
    execution: dict[str, object]


def _model_seed(seed: SeedLike, *, replicate: int, generated: bool) -> int:
    """Separate point and replicate fits without consuming sampling streams."""
    sequence = np.random.SeedSequence(
        seed, spawn_key=(0x505252, int(replicate + 1), int(generated))
    )
    return int(np.random.default_rng(sequence).integers(0, 2**32 - 1))


def run_utility(
    holdout: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    metric: MLE,
    options: ResolvedOptions,
    draw: FixedDraw | None = None,
) -> UtilityExecution:
    """Evaluate predictive utility on the common single-holdout draws.

    Each comparison trains a real predictor and scores it alongside a generated
    predictor on the same real test rows. Reference uncertainty reuses the point
    generated model, without a separate point-scoring pass. Candidate-only and both-side uncertainty resample generated
    rows and refit that predictor for every replicate. Predictor encodings are fitted
    solely on their own training rows; no public argument preparation occurs here.
    """
    assert schema.target is not None
    domain = SourceDomain.prepare(
        {"test": holdout, "generated": generated},
        policy="observed_target",
        columns=(schema.target,),
    )
    holdout, generated = domain.pools["test"], domain.pools["generated"]
    if any(counts["excluded"] for counts in domain.counts.values()):
        draw = None
    real_train = real_test = holdout
    draw = draw or eligible_fixed_draw(
        len(holdout),
        len(generated),
        seed=options["seed"],
        uncertainty=options["uncertainty"],
    )
    point_draw = utility_point_draw(draw)
    n_res = draw.n_res
    task = metric.resolve_task(real_train, schema)
    metadata: UtilityMetadata = {
        "n_holdout": len(holdout),
        "n_real_train": len(real_train),
        "n_real_test": len(real_test),
        "separate_real_pools": False,
        "n_generated": len(generated),
        "n_res": n_res,
        "n_selected_generated": n_res,
        "n_selected_real_train": n_res,
        "n_selected_real_test": n_res,
        "uncertainty": options["uncertainty"],
        "n_replicates": options["n_replicates"] if options["uncertainty"] != "none" else 0,
        "sampling": "uniform_without_replacement",
        "replacement": False,
        "real_roles_disjoint": True,
        "shared_real_scoring": True,
        "source": "observed_target_inputs",
        "generated_training": (
            "resampled" if options["uncertainty"] in {"candidate", "both"} else "fixed"
        ),
        "task": task,
    }
    if n_res < max(1, metric.n_min):
        missing = MLEOutcome(
            generated_score=float("nan"),
            reference_score=float("nan"),
            ratio=float("nan"),
            reason="insufficient eligible sampled rows",
            diagnostics={"insufficient_data": True},
        )
        return UtilityExecution(
            missing,
            (missing,) * metadata["n_replicates"],
            metadata,
            {"implementation": "unavailable", "device": "cpu"},
        )
    from tabgen_eval.execution.runtime.memory import automatic
    from tabgen_eval.execution.xgboost.device import select_device

    predictors = tuple(c for c in schema.categorical if c != schema.target)
    groups = (
        max(int(holdout[schema.target].nunique()), int(generated[schema.target].nunique()))
        if task == "classification"
        else 1
    )
    device = select_device(
        metric,
        rows=2 * n_res,
        numeric=len(set(schema.numeric) - {schema.target}),
        cardinalities=tuple(
            max(int(holdout[c].nunique()), int(generated[c].nunique())) for c in predictors
        ),
        output_groups=groups if groups > 2 else 1,
    )
    resource_metric = metric if automatic(metric, "memory_limit_mb") else None
    from tabgen_eval.preprocessing.predictors import match_predictor_precision

    (real_train, real_test, generated), precision = match_predictor_precision(
        (real_train, real_test, generated), schema, metric.dtype
    )
    unavailable = set(metric.stat_names).issubset(
        metric.not_applicable(real_train, generated, schema)
    )
    insufficient = n_res < metric.n_min

    def fit_model(training: pd.DataFrame, *, replicate: int, generated: bool) -> FittedPredictor:
        return fit_predictor(
            training,
            schema,
            seed=_model_seed(options["seed"], replicate=replicate, generated=generated),
            task=task,
            device=device,
            memory_limit_mb=metric.memory_limit_mb,
            fit_threads=metric.fit_threads,
            dtype=metric.dtype,
            resource_metric=resource_metric,
        )

    def fit_generated(positions: IntArray, replicate: int) -> FittedPredictor:
        return fit_model(generated.iloc[positions], replicate=replicate, generated=True)

    def score_draw(
        selected: UtilityDraw, generated_model: FittedPredictor | None, replicate: int
    ) -> MLEOutcome:
        if unavailable or insufficient:
            return MLEOutcome(
                generated_score=float("nan"),
                reference_score=float("nan"),
                ratio=float("nan"),
                reason="no predictor columns" if unavailable else "insufficient sampled rows",
                diagnostics={
                    "not_applicable": bool(unavailable),
                    "insufficient_data": insufficient,
                },
            )
        assert generated_model is not None
        real_model = fit_model(
            real_train.iloc[selected.real_train], replicate=replicate, generated=False
        )
        return score_mle(generated_model, real_model, real_test.iloc[selected.real_test])

    resample_generated = options["uncertainty"] in {"candidate", "both"}
    generated_model = (
        None
        if unavailable or insufficient or resample_generated
        else fit_generated(point_draw.generated, -1)
    )
    replicates: list[MLEOutcome] = []
    if options["uncertainty"] != "none":
        draws = utility_draws(
            draw,
            n_replicates=options["n_replicates"],
            seed=options["seed"],
            include_generated=options["uncertainty"] in {"candidate", "both"},
        )
        for index, selected in enumerate(draws):
            replicate_model = generated_model
            if options["uncertainty"] in {"candidate", "both"} and not (
                unavailable or insufficient
            ):
                replicate_model = fit_generated(selected.generated, index)
            replicates.append(score_draw(selected, replicate_model, index))
    point = replicates[0] if replicates else score_draw(point_draw, generated_model, -1)
    recorded = metadata.copy()
    recorded["preprocessing"] = precision
    return UtilityExecution(
        point,
        tuple(replicates),
        recorded,
        {
            "implementation": "xgboost",
            "device": device,
            "routing_reason": metric._xgboost_routing_reason,
            "fit_threads": metric.fit_threads,
            "memory_plan": dict(metric._memory_plan),
        },
    )
