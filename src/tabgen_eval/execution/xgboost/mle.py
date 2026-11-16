"""MLE fitting and raw AUROC/MAE scoring for explicitly assigned data roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabgen_eval._typing import Precision, TaskKind
from tabgen_eval.execution.runtime.outcomes import UtilityOutcome
from tabgen_eval.preprocessing.predictors import match_predictor_precision
from tabgen_eval.schema import Schema

from .prediction import PredictorOptions, fit_predictor, labelled_rows, score_predictor
from .utility import MLEOutcome, score_mle

if TYPE_CHECKING:
    from tabgen_eval.metrics import Metric


def outcome_values(outcome: MLEOutcome) -> dict[str, float]:
    """Expose raw score, raw-score ratio, and task prediction comparisons."""
    return {
        "score": outcome.generated_score,
        "ratio": outcome.ratio,
        "prediction_agreement": outcome.prediction_agreement,
        "prediction_mse": outcome.prediction_mse,
    }


def compute_explicit(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    schema: Schema,
    *,
    task: TaskKind,
    device: str = "cpu",
    dtype: Precision = "float64",
    memory_limit_mb: float = 2048,
    rng: np.random.Generator | None = None,
    fit_threads: int = 1,
    baseline_only: bool = False,
    resource_metric: Metric | None = None,
) -> UtilityOutcome:
    """Fit explicit training pools and score on identical real observations.

    Test-only evaluation skips the generated fit while consuming its seed and
    returns the real-trained raw score without creating a ratio benchmark.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    (baseline, reference, candidate), precision = match_predictor_precision(
        (baseline, reference, candidate), schema, dtype
    )
    seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(2)]
    options: PredictorOptions = {
        "task": task,
        "device": device,
        "dtype": dtype,
        "memory_limit_mb": memory_limit_mb,
        "fit_threads": fit_threads,
    }
    if resource_metric is not None:
        options["resource_metric"] = resource_metric
    generated_model = (
        None if baseline_only else fit_predictor(candidate, schema, seed=seeds[0], **options)
    )
    real_model = fit_predictor(baseline, schema, seed=seeds[1], **options)
    if baseline_only:
        assert schema.target is not None
        test, truth = labelled_rows(reference, schema.target, task)
        score, info = score_predictor(real_model, test, truth)
        return UtilityOutcome(
            {},
            {"score": score},
            dict(
                target=schema.target,
                task=task,
                scoring="mae" if task == "regression" else "auroc",
                shared_test_set=True,
                preprocessing=precision,
                fits={"baseline": info},
                usable_rows={"scoring": len(test), "baseline": real_model.training_rows},
            ),
        )
    assert generated_model is not None
    outcome = score_mle(generated_model, real_model, reference)
    diagnostics = dict(
        outcome.diagnostics,
        generated_score=outcome.generated_score,
        reference_score=outcome.reference_score,
        ratio=outcome.ratio,
        reason=outcome.reason,
        preprocessing=precision,
    )
    return UtilityOutcome(outcome_values(outcome), {"score": outcome.reference_score}, diagnostics)
