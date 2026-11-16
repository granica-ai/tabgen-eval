"""Machine Learning Efficacy (MLE): raw AUROC/MAE and generated-to-real ratios."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral, Real

import numpy as np
import pandas as pd

from tabgen_eval._typing import TaskChoice, TaskKind
from tabgen_eval.execution.runtime.outcomes import UtilityOutcome
from tabgen_eval.execution.xgboost.prediction import prediction_task
from tabgen_eval.schema import Schema

from .base import Direction, Metric, StatSpec


class MLE(Metric):
    """Compare generated-trained and real-trained predictive performance.

    Classification reports raw AUROC and generated AUROC / real AUROC; higher
    scores and ratios are better. Regression reports raw mean absolute error
    (MAE) and generated MAE / real MAE; lower scores and ratios are better.
    Ratios use the raw scores without adjustment, capping, or smoothing.
    Positive/zero gives infinity and zero/zero is undefined. Raw generated and
    real scores remain in score.value and score.ideal. MAE is defined for a
    single scoring observation and for constant scoring targets.

    Evaluate selects disjoint real training/scoring draws and a matched generated
    training draw without replacement. Reference repetitions reuse the generated
    model and rescore it on each fresh real test draw. Both repetitions also
    refit that model. Classification agreement and regression prediction MSE
    reuse these predictions without additional fits.

    Records with missing targets are excluded before sampling and uncertainty
    draws. Missing predictors remain available to the training-fitted predictor
    policy. Prediction MSE is the mean squared difference between the two models'
    predictions on their shared real scoring rows; it is distinct from target
    prediction error.

    Schema.target is required and excluded from predictors. Auto task follows
    its declared role, then real-data dtype. Untuned XGBoost models use shared
    training-only CPU/CUDA preprocessing. Auto device prefers a compatible GPU
    when XGBoost, CuPy and memory permit, otherwise CPU. fit_threads=1 budgets CPU
    threads per fit. Evaluation automatically plans preprocessing and estimated
    fitting allowances from available memory; engine allocations are not a hard cap.
    """

    name = "mle"
    category = "utility"
    needs_target = True
    execution_backend = "xgboost"
    diagnostics_group = "utility_diagnostics"
    bootstrapped = True
    task: TaskChoice = "auto"
    _resolved_task: TaskKind
    fit_threads = 1
    memory_limit_mb = 2048
    stats = (
        StatSpec("score", "MLE (Raw Score)", Direction.HIGHER_BETTER, bounds=None),
        StatSpec(
            "ratio",
            "MLE Ratio",
            Direction.HIGHER_BETTER,
            bounds=(0.0, float("inf")),
            target=1.0,
            compare_to_ideal=False,
        ),
        StatSpec(
            "prediction_agreement",
            "Prediction Agreement",
            Direction.HIGHER_BETTER,
            compare_to_ideal=False,
        ),
        StatSpec(
            "prediction_mse",
            "Prediction Mean Squared Difference",
            Direction.LOWER_BETTER,
            bounds=(0.0, float("inf")),
            compare_to_ideal=False,
        ),
    )

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task not in {"auto", "classification", "regression"}:
            raise ValueError("task must be 'auto', 'classification', or 'regression'")
        if (
            isinstance(self.fit_threads, (bool, np.bool_))
            or not isinstance(self.fit_threads, Integral)
            or self.fit_threads < 1
        ):
            raise ValueError("fit_threads must be a positive integer")
        if (
            isinstance(self.memory_limit_mb, (bool, np.bool_))
            or not isinstance(self.memory_limit_mb, Real)
            or not np.isfinite(self.memory_limit_mb)
            or self.memory_limit_mb <= 0
        ):
            raise ValueError("memory_limit_mb must be positive and finite")

    def resolve_task(self, reference: pd.DataFrame, schema: Schema) -> TaskKind:
        """Resolve target role/dtype while preserving an explicit task."""
        return prediction_task(schema, reference, self.task)

    def validate_inputs(
        self, reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema
    ) -> None:
        """Require the target and resolve its raw-score metadata."""
        super().validate_inputs(reference, candidate, schema)
        self._resolved_task = self.resolve_task(reference, schema)
        regression = self._resolved_task == "regression"
        scoring = "MAE" if regression else "AUROC"
        direction = Direction.LOWER_BETTER if regression else Direction.HIGHER_BETTER
        self.stats = (
            replace(
                type(self).stats[0],
                display=f"MLE ({scoring})",
                direction=direction,
                bounds=(0.0, float("inf")) if regression else (0.0, 1.0),
            ),
            replace(type(self).stats[1], display=f"MLE Ratio ({scoring})", direction=direction),
            *type(self).stats[2:],
        )

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        features = (set(schema.numeric) | set(schema.categorical)) - {schema.target}
        if not features:
            return set(self.stat_names)
        if reference is None:
            raise ValueError("MLE applicability requires reference observations")
        task = self.resolve_task(reference, schema)
        return {"prediction_mse" if task == "classification" else "prediction_agreement"}

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> dict[str, float]:
        """Compute one source-seeded point on disjoint real roles."""
        from tabgen_eval._options import resolve_options
        from tabgen_eval.execution.runtime.utility import run_utility
        from tabgen_eval.execution.xgboost.mle import outcome_values

        self.validate_inputs(reference, candidate, schema)
        seed = 42 if rng is None else int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        outcome = run_utility(
            reference,
            candidate,
            schema,
            metric=self,
            options=resolve_options(seed=seed),
        )
        return outcome_values(outcome.point)

    def compute_explicit(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        baseline: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> UtilityOutcome:
        """Score explicit internal pools through the common predictor protocol."""
        return self.compute_explicit_with_budget(
            reference,
            candidate,
            baseline,
            schema,
            rng=rng,
            point=True,
            fit_threads=self.fit_threads,
        )

    def compute_explicit_with_budget(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        baseline: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
        point: bool = False,
        fit_threads: int = 1,
        baseline_only: bool = False,
    ) -> UtilityOutcome:
        """Score internal roles; real-only calibration fits one baseline model."""
        from tabgen_eval.execution.runtime.memory import automatic
        from tabgen_eval.execution.xgboost.mle import compute_explicit

        self.validate_inputs(reference, candidate, schema)
        if fit_threads < 1:
            raise ValueError("fit_threads must be positive")
        from tabgen_eval.execution.xgboost.device import select_device

        predictors = tuple(c for c in schema.categorical if c != schema.target)
        groups = (
            max(int(baseline[schema.target].nunique()), int(candidate[schema.target].nunique()))
            if self._resolved_task == "classification"
            else 1
        )
        device = select_device(
            self,
            rows=len(candidate) + len(baseline),
            numeric=len(set(schema.numeric) - {schema.target}),
            cardinalities=tuple(
                max(int(baseline[c].nunique()), int(candidate[c].nunique())) for c in predictors
            ),
            output_groups=groups if groups > 2 else 1,
        )
        return compute_explicit(
            reference,
            candidate,
            baseline,
            schema,
            task=self._resolved_task,
            device=device,
            dtype=self.dtype,
            memory_limit_mb=self.memory_limit_mb,
            rng=rng,
            fit_threads=fit_threads,
            baseline_only=baseline_only,
            resource_metric=self if automatic(self, "memory_limit_mb") else None,
        )
