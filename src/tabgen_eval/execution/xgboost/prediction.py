"""Training-fitted predictor encoding and reusable supervised scoring.

Encodings learn only from a model's own training rows. XGBoost owns fitted
model state; these helpers prepare predictors, exclude missing targets, and
preserve score diagnostics without sampling or assigning evaluation roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.metrics import roc_auc_score

from tabgen_eval._typing import FloatArray, Precision, TaskChoice, TaskKind
from tabgen_eval.execution.python.utility import regression_mae
from tabgen_eval.models import FitUnavailable, XGBoostModel
from tabgen_eval.preprocessing.categories import CategoryVocabulary, category_display
from tabgen_eval.preprocessing.columns import numeric_values
from tabgen_eval.preprocessing.predictors import PredictorPreprocessor
from tabgen_eval.schema import Schema

if TYPE_CHECKING:
    from tabgen_eval.metrics import Metric


class _RequiredPredictorOptions(TypedDict):
    task: TaskKind
    device: str
    memory_limit_mb: float
    fit_threads: int


class PredictorOptions(_RequiredPredictorOptions, total=False):
    """Shared keyword arguments for fitting independently encoded predictors."""

    dtype: Precision
    resource_metric: Metric


def prediction_task(schema: Schema, reference: pd.DataFrame, task: TaskChoice = "auto") -> TaskKind:
    """Resolve the declared target task, preserving explicit task selection."""
    if task != "auto":
        return task
    if schema.target in schema.categorical:
        return "classification"
    if schema.target in schema.numeric:
        return "regression"
    dtype = reference[schema.target].dtype
    return (
        "regression"
        if pdt.is_numeric_dtype(dtype) and not pdt.is_bool_dtype(dtype)
        else "classification"
    )


def labelled_rows(
    frame: pd.DataFrame, target: str, task: TaskKind
) -> tuple[pd.DataFrame, pd.Series | FloatArray]:
    """Exclude actual missing classification targets or nonfinite numeric targets."""
    values = frame[target]
    if task == "regression":
        values = numeric_values(values).astype(float, copy=False)
        keep = np.flatnonzero(np.isfinite(values))
        return frame.iloc[keep], values[keep]
    keep = np.flatnonzero(values.notna().to_numpy())
    return frame.iloc[keep], values.iloc[keep]


@dataclass
class FittedPredictor:
    """One fitted model and its training-only encoding, reusable across test draws."""

    task: TaskKind
    target: str
    training_rows: int
    model: XGBoostModel | None = None
    encoding: PredictorPreprocessor | None = None
    classes: CategoryVocabulary | None = None
    diagnostics: dict[str, object] = field(default_factory=dict[str, object])
    resource_metric: Metric | None = None
    retained_memory_bytes: int = 0


def _encoding_bytes(encoding: PredictorPreprocessor) -> int:
    """Estimate retained transform tensors and host vocabulary metadata."""
    tensors = [
        value
        for state in (encoding.numeric_state, encoding.category_state)
        for value in vars(state).values()
    ]
    return sum(int(value.numel() * value.element_size()) for value in tensors) + sum(
        128 * len(vocabulary) for vocabulary in encoding.vocabularies
    )


def fit_predictor(
    training: pd.DataFrame,
    schema: Schema,
    *,
    task: TaskKind,
    device: str,
    memory_limit_mb: float,
    seed: int,
    fit_threads: int,
    dtype: Precision = "float64",
    resource_metric: Metric | None = None,
) -> FittedPredictor:
    """Fit on an explicit device, retaining data failures as diagnostics."""
    device = XGBoostModel.execution_device(device)
    XGBoostModel.load_backend()
    assert schema.target is not None
    train, target = labelled_rows(training, schema.target, task)
    result = FittedPredictor(task, schema.target, len(train))
    result.resource_metric = resource_metric
    result.diagnostics = {
        "training_rows": len(train),
        "fit_threads": int(fit_threads),
        "random_state": int(seed),
    }
    try:
        if resource_metric is not None:
            from tabgen_eval.execution.runtime.memory import plan_metric_memory

            plan = plan_metric_memory(resource_metric, device)
            memory_limit_mb = resource_metric.memory_limit_mb
            result.diagnostics["preparation_memory_plan"] = plan.to_dict()
        if not len(train):
            raise FitUnavailable("missing_target")
        if task == "classification":
            classes = CategoryVocabulary.fit(target)
            if all(isinstance(label, str) for label in classes.values):
                classes = CategoryVocabulary.fit(pd.Series(sorted(classes.values), dtype=object))
            result.classes = classes
            if len(classes.values) < 2:
                raise FitUnavailable("classification_class_support")
            target = classes.transform(target)
            result.diagnostics["classes"] = [category_display(label) for label in classes.values]
        result.encoding, design = PredictorPreprocessor.fit_transform(
            train,
            schema,
            memory_limit_mb * 1024**2,
            device=device,
            dtype=dtype,
        )
        if resource_metric is not None:
            state_bytes = _encoding_bytes(result.encoding)
            design_bytes = int(design.numel() * design.element_size())
            plan = plan_metric_memory(
                resource_metric,
                device,
                owned_task_bytes=state_bytes + design_bytes,
            )
            from tabgen_eval.execution.runtime.memory import (
                xgboost_model_bytes,
                xgboost_workspace_bytes,
            )

            output_groups = (
                len(result.classes) if task == "classification" and len(result.classes) > 2 else 1
            )
            estimate = xgboost_workspace_bytes(
                len(train),
                result.encoding.feature_types.count("q"),
                result.encoding.categorical_cardinalities,
                output_groups=output_groups,
            )
            result.diagnostics["memory_plan"] = plan.to_dict()
            result.diagnostics["engine_workspace_estimate_bytes"] = int(estimate)
            result.diagnostics["engine_workspace_is_estimated"] = True
            result.retained_memory_bytes = state_bytes + xgboost_model_bytes(output_groups)
            if estimate > plan.task_workspace_bytes:
                raise FitUnavailable(
                    "xgboost_workspace_limit",
                    estimated_bytes=int(estimate),
                    memory_limit_bytes=plan.task_workspace_bytes,
                )
        result.model = XGBoostModel(task=task, device=device).fit(
            design,
            target,
            seed=seed,
            fit_threads=fit_threads,
            feature_types=result.encoding.feature_types,
            feature_names=result.encoding.feature_names,
            categorical_cardinalities=result.encoding.categorical_cardinalities,
        )
        result.diagnostics.update(
            result.model.diagnostics,
            features=design.shape[1],
            preprocessing=result.encoding.diagnostics,
        )
    except (FitUnavailable, FloatingPointError, MemoryError) as error:
        result.model = None
        result.diagnostics.update(getattr(error, "diagnostics", {}))
        result.diagnostics.update(
            converged=False,
            reason=getattr(
                error,
                "reason",
                "memory_limit" if isinstance(error, MemoryError) else "nonfinite_fit",
            ),
        )
    return result


@dataclass(frozen=True)
class PredictorPrediction:
    """Predictions and diagnostics from one encoded scoring pool.

    Classification predictions are probability columns in the fitted model's
    own typed class order. Missing output denotes an unavailable prediction.
    """

    values: FloatArray | None
    diagnostics: dict[str, object]


def predict_predictor(predictor: FittedPredictor, test: pd.DataFrame) -> PredictorPrediction:
    """Predict once for reuse by supervised scores and model agreement."""
    info = dict(predictor.diagnostics, scoring_rows=len(test))
    if predictor.model is None:
        return PredictorPrediction(None, info)
    try:
        if not len(test):
            raise FitUnavailable("missing_target")
        encoding = predictor.encoding
        assert encoding is not None
        if predictor.resource_metric is not None:
            from tabgen_eval.execution.runtime.memory import plan_metric_memory

            plan = plan_metric_memory(
                predictor.resource_metric,
                encoding.device,
                owned_task_bytes=predictor.retained_memory_bytes,
            )
            encoding = replace(encoding, memory_limit=plan.task_workspace_bytes)
            info["scoring_memory_plan"] = plan.to_dict()
        # This adapter requests probabilities or regression values, both floating.
        values = cast(
            FloatArray,
            np.asarray(
                predictor.model.predict(
                    encoding.transform(test),
                    prediction="probability" if predictor.task == "classification" else "value",
                )
            ),
        )
        if not np.isfinite(values).all():
            raise FitUnavailable("nonfinite_predictions")
        return PredictorPrediction(values, info)
    except (FitUnavailable, FloatingPointError, MemoryError) as error:
        info.update(getattr(error, "diagnostics", {}))
        info.update(
            converged=False,
            reason=getattr(
                error,
                "reason",
                "memory_limit" if isinstance(error, MemoryError) else "nonfinite_score",
            ),
        )
        return PredictorPrediction(None, info)


def score_predictions(
    predictor: FittedPredictor, prediction: PredictorPrediction, target: pd.Series | FloatArray
) -> tuple[float, dict[str, object]]:
    """Score reusable predictions as MAE or binary/macro one-vs-rest AUROC."""
    info = dict(prediction.diagnostics)
    if prediction.values is None:
        return float("nan"), info
    try:
        if predictor.task == "classification":
            assert predictor.classes is not None
            truth = predictor.classes.transform(target)
            count = len(predictor.classes.values)
            if (truth < 0).any() or len(np.unique(truth)) != count:
                raise FitUnavailable("classification_class_support")
        else:
            truth = np.asarray(target)
        if predictor.task == "regression":
            score = regression_mae(truth, prediction.values)
            if not np.isfinite(score):
                info["score_reason"] = (
                    "undefined_mae_no_scoring_rows"
                    if not len(truth)
                    else "undefined_mae_nonfinite_score"
                )
        elif count == 2:
            score = float(roc_auc_score(truth, prediction.values[:, 1]))
        else:
            score = float(
                roc_auc_score(truth, prediction.values, multi_class="ovr", average="macro")
            )
        return score, info
    except (FitUnavailable, FloatingPointError) as error:
        info.update(getattr(error, "diagnostics", {}))
        info.update(converged=False, reason=getattr(error, "reason", "nonfinite_score"))
        return float("nan"), info


def score_predictor(
    predictor: FittedPredictor, test: pd.DataFrame, target: pd.Series | FloatArray
) -> tuple[float, dict[str, object]]:
    """Score one already-labelled test draw without refitting the model."""
    return score_predictions(predictor, predict_predictor(predictor, test), target)
