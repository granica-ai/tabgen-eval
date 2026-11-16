"""Custom classifier fold execution and the shared XGBoost fitting adapter.

Functions accept explicit data, options, and a fit callback. Training partitions
fit all preprocessing and models; scoring partitions provide held-out predictions.
Source folds and result containers use the shared execution contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tabgen_eval._typing import FloatArray, IntArray, Precision
from tabgen_eval.execution.classifier.protocol import ClassifierFit
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.models import FitUnavailable, XGBoostModel
from tabgen_eval.schema import Schema

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.preprocessing.sources import PreparedSources


def compute_partitioned(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    reference_folds: IntArray,
    candidate_folds: IntArray,
    *,
    seed: int,
    point: bool = False,
    fit_threads: int = 1,
    cv_folds: int,
    memory_limit_mb: float,
    dtype: Precision = "float64",
    fit_predict: ClassifierFit,
) -> ClassifierOutcome:
    """Fit and score supplied source folds through an adapter's fit callback.

    DataFrames and fold arrays are read-only. Each fold fits training-only transforms
    under the preprocessing byte guard. The callback receives seed+fold, the point
    flag, and the thread budget. Return a backend-neutral outcome containing mean
    clipped AUC and per-fold fit diagnostics. Recognized unusable fits yield NaN;
    dependency and programming errors propagate.
    """
    if fit_threads < 1:
        raise ValueError("fit_threads must be positive")
    from tabgen_eval.preprocessing import prepare_sources

    sources = prepare_sources(reference, candidate, schema, dtype=dtype)
    return compute_prepared(
        sources.reference,
        sources.candidate,
        reference_folds,
        candidate_folds,
        sources=sources,
        seed=seed,
        point=point,
        fit_threads=fit_threads,
        cv_folds=cv_folds,
        memory_limit_mb=memory_limit_mb,
        fit_predict=fit_predict,
    )


def compute_prepared(
    reference: TensorTable,
    candidate: TensorTable,
    reference_folds: IntArray,
    candidate_folds: IntArray,
    *,
    sources: PreparedSources,
    seed: int,
    point: bool,
    fit_threads: int,
    cv_folds: int,
    memory_limit_mb: float,
    fit_predict: ClassifierFit,
) -> ClassifierOutcome:
    """Fit a custom classifier over already precision-matched typed source draws.

    Paired real-generated and real-real comparisons reuse the evaluation's
    source precision and category identities. Only fold-fitted state is rebuilt.
    """
    if fit_threads < 1:
        raise ValueError("fit_threads must be positive")
    from tabgen_eval.preprocessing import fit_fold

    values, diagnostics = [], []
    for fold in range(cv_folds):
        data = train = test = ytrain = ytest = scores = None
        try:
            data = fit_fold(
                reference,
                candidate,
                reference_folds,
                candidate_folds,
                fold,
                cardinalities=sources.cardinalities,
                numeric_names=sources.numeric_names,
                categorical_names=sources.categorical_names,
                category_values=sources.category_values,
                policy="linear",
                memory_limit_bytes=memory_limit_mb * 1024**2,
            )
            train, test, ytrain, ytest = [
                value.numpy()
                for value in (data.train, data.test, data.training_labels, data.scoring_labels)
            ]
            scores, info = fit_predict(
                train,
                ytrain,
                test,
                seed=seed + fold,
                point=point,
                fit_threads=fit_threads,
            )
            if not np.isfinite(scores).all():
                raise FitUnavailable()
            values.append(max(float(roc_auc_score(ytest, scores)), 0.5))
            info.update(
                fold=fold,
                training_rows=len(train),
                scoring_rows=len(test),
                features=train.shape[1],
                preprocessing=dict(data.diagnostics, precision=dict(sources.precision)),
            )
            diagnostics.append(info)
        except (FitUnavailable, FloatingPointError) as error:
            values.append(np.nan)
            diagnostics.append(
                {
                    **getattr(error, "diagnostics", {}),
                    "fold": fold,
                    "converged": False,
                    "reason": getattr(error, "reason", "preprocessing_or_memory_limit"),
                }
            )
        finally:
            # NumPy views and score views can retain a fold's tensor storage.
            # Release every owner before the next fold's memory preflight.
            data = train = test = ytrain = ytest = scores = None
    return ClassifierOutcome(float(np.mean(values)), diagnostics)


def nonlinear_fit_predict(
    train: FloatArray,
    labels: FloatArray,
    test: FloatArray,
    *,
    device: str,
    seed: int,
    fit_threads: int = 1,
) -> tuple[FloatArray, dict[str, object]]:
    """Delegate an untuned nonlinear detector fit to the shared XGBoost owner.

    Return held-out margins and training diagnostics. Encoded inputs are
    read-only; model lifecycle, device validation, and engine failures belong
    to models.XGBoostModel, independent of metric classes.
    """
    return XGBoostModel(device=device).fit_predict(
        train, labels, test, seed=seed, fit_threads=fit_threads, prediction="margin"
    )
