"""MLE raw-score ratios and shared prediction comparisons for CPU/CUDA models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tabgen_eval._typing import TaskKind
from tabgen_eval.execution.python.agreement import classification_agreement, regression_disagreement
from tabgen_eval.schema import Schema

from .prediction import (
    FittedPredictor,
    PredictorOptions,
    fit_predictor,
    labelled_rows,
    predict_predictor,
    score_predictions,
)


@dataclass(frozen=True)
class MLEOutcome:
    """Raw predictive scores, their ratio, and computation context.

    Regression uses mean absolute error; classification uses AUROC. The ratio
    divides the generated-trained score by the reference-trained score directly.
    ``reason`` explains an undefined ratio;
    positive infinity is a defined result and has no failure reason.
    Task-specific prediction agreement uses these same model predictions,
    without additional fitting or prediction calls.
    No sampling, intervals, calibration flags, or quality labels are included.
    """

    generated_score: float
    reference_score: float
    ratio: float
    reason: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict[str, object])
    prediction_agreement: float = float("nan")
    prediction_mse: float = float("nan")


def _score_ratio(generated_score: float, reference_score: float, *, scoring: str) -> MLEOutcome:
    """Divide valid raw scores without smoothing, shifting or clipping."""
    generated, reference = float(generated_score), float(reference_score)
    valid = np.isfinite(generated) and np.isfinite(reference)
    if scoring == "auroc":
        valid = valid and 0 <= generated <= 1 and 0 <= reference <= 1
        undefined = "undefined classification AUROC"
    else:
        valid = valid and generated >= 0 and reference >= 0
        undefined = "undefined regression MAE"
    reason = None
    if not valid:
        ratio, reason = float("nan"), undefined
    elif reference > 0:
        ratio = generated / reference
    elif generated > 0:
        ratio = float("inf")
    else:
        ratio = float("nan")
        reason = "zero generated and reference scores"
    return MLEOutcome(generated, reference, ratio, reason)


def regression_ratio(generated_mae: float, reference_mae: float) -> MLEOutcome:
    """Compare raw MAE without caps or denominator smoothing.

    Lower ratios mean less generated-trained prediction error. Positive error
    over zero reference error gives infinity; zero divided by zero is undefined.
    Negative or nonfinite errors are invalid scores.
    """
    return _score_ratio(generated_mae, reference_mae, scoring="mae")


def classification_ratio(generated_auroc: float, reference_auroc: float) -> MLEOutcome:
    """Compare raw generated-trained and reference-trained AUROC directly.

    Chance and below-chance scores remain valid observed AUROCs. Positive AUROC
    over zero reference AUROC gives infinity; zero divided by zero is undefined.
    Nonfinite scores or values outside AUROC's [0, 1] range are undefined.
    """
    return _score_ratio(generated_auroc, reference_auroc, scoring="auroc")


def score_mle(
    generated_model: FittedPredictor, reference_model: FittedPredictor, real_test: pd.DataFrame
) -> MLEOutcome:
    """Compare two fitted models on the same target-filtered real test rows."""
    task = generated_model.task
    test, truth = labelled_rows(real_test, generated_model.target, task)
    generated_prediction = predict_predictor(generated_model, test)
    reference_prediction = predict_predictor(reference_model, test)
    generated, generated_info = score_predictions(generated_model, generated_prediction, truth)
    reference, reference_info = score_predictions(reference_model, reference_prediction, truth)
    agreement = (
        classification_agreement(
            generated_prediction.values,
            reference_prediction.values,
            generated_model.classes,
            reference_model.classes,
        )
        if task == "classification"
        else regression_disagreement(generated_prediction.values, reference_prediction.values)
    )
    ratio = regression_ratio if task == "regression" else classification_ratio
    outcome = ratio(generated, reference)
    return MLEOutcome(
        outcome.generated_score,
        outcome.reference_score,
        outcome.ratio,
        outcome.reason,
        {
            "target": generated_model.target,
            "task": task,
            "scoring": "mae" if task == "regression" else "auroc",
            "shared_test_set": True,
            "usable_rows": {
                "scoring": len(test),
                "generated": generated_model.training_rows,
                "reference": reference_model.training_rows,
            },
            "fits": {"generated": generated_info, "reference": reference_info},
        },
        prediction_agreement=agreement if task == "classification" else float("nan"),
        prediction_mse=agreement if task == "regression" else float("nan"),
    )


def compute_mle(
    real_test: pd.DataFrame,
    generated_train: pd.DataFrame,
    real_train: pd.DataFrame,
    schema: Schema,
    *,
    task: TaskKind,
    device: str = "cpu",
    memory_limit_mb: float = 1024,
    rng: np.random.Generator | None = None,
    fit_threads: int = 1,
) -> MLEOutcome:
    """Fit generated and real models once and score on identical real test rows.

    Pools are explicit and are not resampled or partitioned here. Each model
    learns predictor encodings solely from its own training pool. Actual missing
    targets are excluded; the remaining real test rows are shared by both fits.
    Regression scores raw MAE; classification scores binary or macro
    one-versus-rest AUROC. The ratio divides the generated-trained score by the
    real-trained score without transforming either score.
    The caller supplies the
    target role, task, execution device, and fit budgets.
    """
    if schema.target is None:
        raise ValueError("MLE requires an explicit schema target")
    if fit_threads < 1:
        raise ValueError("fit_threads must be positive")
    if not np.isfinite(memory_limit_mb) or memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive and finite")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    rng = np.random.default_rng(0) if rng is None else rng
    from tabgen_eval.preprocessing.predictors import match_predictor_precision

    (real_train, generated_train, real_test), _ = match_predictor_precision(
        (real_train, generated_train, real_test), schema
    )
    options: PredictorOptions = {
        "task": task,
        "device": device,
        "memory_limit_mb": memory_limit_mb,
        "fit_threads": fit_threads,
    }
    generated_model = fit_predictor(
        generated_train, schema, seed=int(rng.integers(0, 2**32 - 1)), **options
    )
    reference_model = fit_predictor(
        real_train, schema, seed=int(rng.integers(0, 2**32 - 1)), **options
    )
    return score_mle(generated_model, reference_model, real_test)
