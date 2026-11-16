"""Independent scikit-learn linear C2ST references used only by tests.

The native solvers are checked against a separate optimizer. Fold preparation
and callback lifecycle use the shared production contracts.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from tabgen_eval._typing import FloatArray, IntArray
from tabgen_eval.execution.python.classifier import compute_partitioned as compute_classifier
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.schema import Schema

if TYPE_CHECKING:
    from tabgen_eval.metrics.c2st import LinearC2ST, LinearOptions


def linear_fit_predict(
    train: FloatArray,
    labels: FloatArray,
    test: FloatArray,
    *,
    options: LinearOptions,
    ridge: float | None,
    seed: int,
    fit_threads: int = 1,
) -> tuple[FloatArray, dict[str, object]]:
    """Fit ridge logistic regression with scikit-learn's L-BFGS or SAGA solver.

    Read-only dense arrays contain training-fitted features. The options mapping
    supplies feature/work routing estimates, solver budgets and one tolerance.
    ``direct`` selects L-BFGS, ``sketchysaga`` selects unpreconditioned SAGA, and
    ``auto`` uses SAGA above the feature/work cutoffs. No dense Hessian is formed.
    ``ridge=None`` uses 1/n in the average-loss objective. L-BFGS records an exact
    post-fit gradient diagnostic; convergence requires that diagnostic to meet
    the threshold without a convergence warning. SAGA retains scikit-learn's
    coefficient-change stopping rule without an extra full-gradient evaluation.
    The reference optimizes in FP64 over the same prepared values. Upcasting
    encoded FP32 observations preserves their values without refitting transforms,
    while avoiding sklearn's FP32 prediction rounding inside L-BFGS. The caller
    controls numerical-library threads.
    """
    train = np.asarray(train, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    ridge = 1 / len(train) if ridge is None else ridge
    d = train.shape[1] + 1
    lbfgs = options["solver"] == "direct" or (
        options["solver"] == "auto"
        and train.shape[1] <= options["direct_max_columns"]
        and len(train) * d**2 <= options["direct_max_work"]
    )
    model = LogisticRegression(
        solver="lbfgs" if lbfgs else "saga",
        C=1 / (len(train) * ridge),
        max_iter=options["max_iter"] if lbfgs else options["saga_max_epochs"],
        tol=options["tolerance"] / np.sqrt(d) if lbfgs else options["tolerance"],
        random_state=seed % (2**32 - 1),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train, labels)
    from scipy.special import expit

    iterations = int(np.max(model.n_iter_))
    fit_warning = any(issubclass(w.category, ConvergenceWarning) for w in captured)
    if not lbfgs:
        return model.decision_function(test), {
            "solver": "python-saga",
            "optimizer_dtype": "float64",
            "tolerance": options["tolerance"],
            "converged": not fit_warning,
            "iterations": iterations,
            "epochs_completed": iterations,
            "stopping_rule": "sklearn_coefficient_change",
            "gradient_squared": None,
            "online_gradient_squared": None,
            "online_gradient_norm": None,
            "online_gradient_infinity": None,
            "initial_online_gradient_infinity": None,
            "gradient_stopping_threshold": None,
            "full_gradient_evaluations": 0,
            "decrement_squared": None,
            "effective_ridge": ridge,
            "reason": "max_epochs" if fit_warning else "coefficient_change",
        }
    logits = model.decision_function(train)
    residual = np.where(labels == 1, -expit(-logits), expit(logits))
    gradient = train.T @ residual / len(train) + ridge * model.coef_[0]
    gradient_squared = float(gradient @ gradient + np.mean(residual) ** 2)
    converged = (
        not fit_warning
        and np.isfinite(gradient_squared)
        and gradient_squared <= options["tolerance"] ** 2
    )
    reason = "gradient_squared" if converged else "not_converged"
    if fit_warning:
        reason = "sklearn_convergence_warning"
    return model.decision_function(test), {
        "solver": "python-lbfgs",
        "optimizer_dtype": "float64",
        "tolerance": options["tolerance"],
        "converged": bool(converged),
        "iterations": iterations,
        "gradient_squared": gradient_squared if np.isfinite(gradient_squared) else None,
        "decrement_squared": None,
        "effective_ridge": ridge,
        "stopping_rule": "gradient_squared",
        "convergence_warning": fit_warning,
        "reason": reason,
    }


def fit_predict(
    metric: LinearC2ST,
    train: FloatArray,
    labels: FloatArray,
    test: FloatArray,
    *,
    seed: int,
    point: bool = False,
    fit_threads: int = 1,
) -> tuple[FloatArray, dict[str, object]]:
    """Resolve one metric fixture's options for the independent sklearn fit."""
    return linear_fit_predict(
        train,
        labels,
        test,
        options=metric.native_options(point=point),
        ridge=metric.ridge,
        seed=seed,
        fit_threads=fit_threads,
    )


def compute_partitioned(
    metric: LinearC2ST,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    reference_folds: IntArray,
    candidate_folds: IntArray,
    *,
    seed: int,
    point: bool = False,
    fit_threads: int = 1,
) -> ClassifierOutcome:
    """Score explicit source folds with a test-only sklearn fitting callback."""

    def reference_fit(train, labels, test, *, seed, point, fit_threads=1):
        return fit_predict(
            metric, train, labels, test, seed=seed, point=point, fit_threads=fit_threads
        )

    return compute_classifier(
        reference,
        candidate,
        schema,
        reference_folds,
        candidate_folds,
        seed=seed,
        point=point,
        fit_threads=fit_threads,
        cv_folds=metric.cv_folds,
        memory_limit_mb=metric.memory_limit_mb,
        dtype=metric.dtype,
        fit_predict=reference_fit,
    )
