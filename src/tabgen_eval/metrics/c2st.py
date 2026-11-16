"""Classifier two-sample tests with source-stable folds and held-out scoring.

Each fold's ROC AUC is clipped below at 0.5, and the reported statistic is the
mean of those fold scores: 0.5 means chance-level discrimination, while larger
values mean the detector distinguishes real from generated rows more readily.
Transforms and optimizer decisions are fitted inside training partitions.
Linear C2ST standardizes numeric columns and retains full categorical features.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from tabgen_eval._typing import FloatArray, IntArray
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.python import classifier as classifier_callbacks
from tabgen_eval.execution.runtime.dispatch import context_for, supports
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.schema import Schema

from .base import Direction, Metric, MetricFamily, StatSpec


class LinearOptions(TypedDict):
    """Resolved linear-classifier controls passed to numerical backends."""

    n_folds: int
    solver: Literal["auto", "direct", "sketchysaga"]
    ridge: float
    tolerance: float
    max_iter: int
    saga_max_epochs: int
    saga_batch_size: int
    saga_rank: int
    saga_backoff: bool
    saga_sampled_preparation: bool
    saga_preparation_rows: int
    direct_max_columns: int
    direct_max_work: float
    memory_limit_mb: float


class ClassifierMetric(Metric):
    """Shared cross-validation controls for linear and nonlinear detectors.

    ``cv_folds=3`` accepts 2 through 16 folds. Each comparison assigns unique sampled source rows to folds, and models
    and transformations are refitted within training partitions. Automatic memory
    allowances guard preprocessing/design allocations and native linear fits;
    private constructor budgets support backend tests. ``bootstrap_fit_threads=1`` accepts 1 through 4 CPU threads
    per bootstrap fit. Shared device/precision options are described by Metric.

    Numeric and categorical preprocessing are fixed parts of each detector.
    Categorical predictors use both training classes to fit their vocabulary. Estimators
    supply source fold labels; classifier adapters implement ``_fit_predict``
    and may specialize encoding.
    """

    family = MetricFamily.ESTIMATOR_BASED
    category = "detection"
    cv_folds = 3
    bootstrap_fit_threads = 1
    memory_limit_mb = 2048

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if not 2 <= self.cv_folds <= 16:
            raise ValueError("cv_folds must be in 2..=16")
        if not 1 <= self.bootstrap_fit_threads <= 4 or self.memory_limit_mb <= 0:
            raise ValueError(
                "bootstrap_fit_threads must be in 1..=4; memory_limit_mb must be positive"
            )

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> dict[str, float]:
        """Compute one three-fold comparison; warnings expose nonconverged fits."""
        rng = np.random.default_rng(0) if rng is None else rng
        if min(len(reference), len(candidate)) < self.cv_folds:
            return {self.stat_names[0]: np.nan}
        rf, sf = source_folds(len(reference), len(candidate), self.cv_folds, rng)
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        native = context_for(self, reference, candidate, schema)
        if native is not None:
            outcomes = native.compute_classifier(
                self,
                np.arange(len(reference))[None, :],
                np.arange(len(candidate))[None, :],
                rf[None, :],
                sf[None, :],
                [seed],
                point=True,
            )
            outcome = outcomes[0]
        else:
            outcome = self.compute_partitioned(
                reference, candidate, schema, rf, sf, seed=seed, point=True
            )
        if not outcome.converged and np.isfinite(outcome.value):
            warnings.warn(
                "C2ST classifier did not satisfy the convergence criterion",
                ConvergenceWarning,
                stacklevel=2,
            )
        return {self.stat_names[0]: outcome.value}

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        return set(self.stat_names) if not schema.numeric and not schema.categorical else set()

    def compute_partitioned(
        self,
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
        """Score explicit source folds through native or custom classifier execution."""
        if supports(self):
            context = context_for(self, reference, candidate, schema, n_jobs=fit_threads)
            assert context is not None
            return context.compute_classifier(
                self,
                np.arange(len(reference))[None],
                np.arange(len(candidate))[None],
                np.asarray(reference_folds)[None],
                np.asarray(candidate_folds)[None],
                [seed],
                point=point,
                fit_threads=fit_threads,
            )[0]
        return classifier_callbacks.compute_partitioned(
            reference,
            candidate,
            schema,
            reference_folds,
            candidate_folds,
            seed=seed,
            point=point,
            fit_threads=fit_threads,
            cv_folds=self.cv_folds,
            memory_limit_mb=self.memory_limit_mb,
            dtype=self.dtype,
            fit_predict=self._fit_predict,
        )

    def _fit_predict(
        self,
        train: FloatArray,
        labels: FloatArray,
        test: FloatArray,
        *,
        seed: int,
        point: bool,
        fit_threads: int = 1,
    ) -> tuple[FloatArray, dict[str, object]]:
        """Return decision scores and JSON-compatible fit diagnostics.

        Adapters must accept ``fit_threads`` (default one), the per-fit CPU
        budget. The estimator chooses it for point fits and bootstrap refits.
        Evaluation uses the point solver policy for both fixed and repeated comparisons.
        Raise ``FitUnavailable`` for a noncomputable fit; dependency and
        programmer errors propagate to the caller.
        """
        raise NotImplementedError("Custom classifier metrics must implement _fit_predict")


class LinearC2ST(ClassifierMetric):
    """Detect distribution differences with ridge logistic regression.

    The ``auc`` statistic averages held-out fold AUCs, each clipped to at least
    0.5; lower is better. The real-real comparison provides a finite-sample
    reference when bootstrap estimation is requested. Shared cross-validation
    and memory controls are documented on ClassifierMetric.

    Main options
    ------------
    On native CPU/CUDA backends, ``solver="auto"`` chooses dense direct Newton
    or compact SketchySAGA from encoded feature width, estimated work, and memory
    limits. ``"direct"`` and ``"sketchysaga"`` force a solver route.
    ``ridge=None`` uses 1/n in each training
    fold's average logistic loss using its training-row count; an explicit ridge
    must be finite and positive. The intercept is unpenalized.
    Numeric columns always use guarded training-only standardization and mean
    imputation with missingness indicators. Full categorical indicators are
    unscaled, using the union of both training classes' observed categories.
    Torch CPU and CUDA fits accept ``dtype="float32"`` or ``"float64"``;
    designs, coefficients, and solver workspaces retain the selected precision.

    Convergence and iteration budgets
    ---------------------------------
    Both native solvers use the absolute ``tolerance=1e-4``. Direct Newton stops
    when the full penalized gradient's L2 norm is at most ``tolerance``; its
    squared-gradient diagnostic is compared against ``tolerance**2`` internally.
    SketchySAGA stops when the unpreconditioned online gradient's infinity norm
    is at most ``tolerance``, with no normalization by the initial gradient.
    Newton uses ``max_iter=10000`` and SketchySAGA uses ``saga_max_epochs=10000``.
    Nonconverged fits remain visible in diagnostics.

    Advanced options
    ----------------
    ``saga_batch_size=4096`` accepts 1 through 4096 and is capped at training rows;
    ``saga_rank=64`` is capped at feature width. ``saga_backoff=True`` enables a
    fixed training-probe guard with bounded rate halving. Coherent curvature
    sampling, ``saga_sampled_preparation=True``, starts only at 500,000 combined
    training rows; ``saga_preparation_rows=4096`` controls that sample size.
    Disable it to use all curvature rows.

    Automatic dense cutoffs are ``direct_max_columns=500`` and
    ``direct_max_work=5e10`` for bootstrap fits; point fits use
    ``point_direct_max_columns=1000`` and ``point_direct_max_work=2e11``. These
    positive work/column limits still respect memory limits. Evaluation uses
    point-fit cutoffs for both fixed and repeated comparisons. See ``docs/usage.md``
    for the full option table and solver diagnostics. Native CPU/CUDA fits use
    private RNG streams.
    """

    name = "linear_c2st"
    stats = (StatSpec("auc", "Linear C2ST (AUC)", Direction.LOWER_BETTER, bounds=(0.5, 1.0)),)
    solver = "auto"
    ridge = None
    tolerance = 1e-4
    max_iter = 10000
    saga_max_epochs = 10000
    saga_batch_size = 4096
    saga_rank = 64
    saga_backoff = True
    saga_sampled_preparation = True
    saga_preparation_rows = 4096
    direct_max_columns = 500
    direct_max_work = 5e10
    point_direct_max_columns = 1000
    point_direct_max_work = 2e11

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.solver not in {"auto", "direct", "sketchysaga"}:
            raise ValueError("solver must be auto, direct, or sketchysaga")
        if self.ridge is not None and (
            isinstance(self.ridge, (bool, np.bool_))
            or not isinstance(self.ridge, Real)
            or not np.isfinite(self.ridge)
            or self.ridge <= 0
        ):
            raise ValueError("ridge must be None or finite and positive")
        for name in [
            "tolerance",
            "direct_max_work",
            "point_direct_max_work",
        ]:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.saga_backoff, (bool, np.bool_)):
            raise ValueError("saga_backoff must be boolean")
        self.saga_backoff = bool(self.saga_backoff)
        if not isinstance(self.saga_sampled_preparation, (bool, np.bool_)):
            raise ValueError("saga_sampled_preparation must be boolean")
        self.saga_sampled_preparation = bool(self.saga_sampled_preparation)
        for name in [
            "max_iter",
            "saga_max_epochs",
            "saga_batch_size",
            "saga_rank",
            "saga_preparation_rows",
            "direct_max_columns",
            "point_direct_max_columns",
        ]:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if self.saga_batch_size > 4096:
            raise ValueError("saga_batch_size must be at most 4096")

    def native_options(self, *, point: bool = False) -> LinearOptions:
        """One objective and tolerance; independent point/replicate work cutoffs."""
        return dict(
            n_folds=self.cv_folds,
            solver=self.solver,
            ridge=0.0 if self.ridge is None else float(self.ridge),
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            saga_max_epochs=self.saga_max_epochs,
            saga_batch_size=self.saga_batch_size,
            saga_rank=self.saga_rank,
            saga_backoff=self.saga_backoff,
            saga_sampled_preparation=self.saga_sampled_preparation,
            saga_preparation_rows=self.saga_preparation_rows,
            direct_max_columns=self.point_direct_max_columns if point else self.direct_max_columns,
            direct_max_work=self.point_direct_max_work if point else self.direct_max_work,
            memory_limit_mb=self.memory_limit_mb,
        )


class NonLinearC2ST(ClassifierMetric):
    """Detect nonlinear distribution differences with cross-validated XGBoost.

    The ``auc`` statistic averages held-out fold AUCs clipped at 0.5; chance-level
    0.5 is best, and larger scores indicate easier real/generated discrimination.
    Uses XGBClassifier defaults with histogram construction, no tuning, and no
    early stopping; device, fit threads, and seed are supplied explicitly.

    ClassifierMetric documents ``cv_folds=3``, automatic memory allowances, and
    ``bootstrap_fit_threads=1``. Numerical features are unscaled and retain
    missing values. Categories use native category-versus-rest splits, with
    an explicit missing level and a vocabulary fitted on both training classes.
    XGBoost >=2.1 is required. ``device="auto"`` prefers a compatible GPU when
    XGBoost, CuPy and memory permit, otherwise CPU. Explicit CUDA requests require
    a CUDA-enabled XGBoost installation and compatible CuPy.
    Effective settings/version are recorded per fit. ``converged`` means its
    fixed boosting budget completed, not that an optimization tolerance was met.
    """

    name = "nonlinear_c2st"
    stats = (StatSpec("auc", "Nonlinear C2ST (AUC)", Direction.LOWER_BETTER, bounds=(0.5, 1.0)),)
    execution_backend = "xgboost"

    def compute_partitioned(
        self,
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
        """Use the same native categorical preparation for explicit fold calls."""
        from tabgen_eval.execution.runtime.dispatch import supports

        if not supports(self):
            return super().compute_partitioned(
                reference,
                candidate,
                schema,
                reference_folds,
                candidate_folds,
                seed=seed,
                point=point,
                fit_threads=fit_threads,
            )
        context = context_for(self, reference, candidate, schema, n_jobs=fit_threads)
        return context.compute_classifier(
            self,
            np.arange(len(reference))[None],
            np.arange(len(candidate))[None],
            np.asarray(reference_folds)[None],
            np.asarray(candidate_folds)[None],
            [seed],
            point=point,
        )[0]

    def _fit_predict(
        self,
        train: FloatArray,
        labels: FloatArray,
        test: FloatArray,
        *,
        seed: int,
        point: bool,
        fit_threads: int = 1,
    ) -> tuple[FloatArray, dict[str, object]]:
        """Fit the XGBoost base method with the supplied numerical thread budget."""
        from tabgen_eval.execution.xgboost.device import select_device

        device = select_device(
            self, rows=len(train) + len(test), numeric=train.shape[1], cardinalities=()
        )
        return classifier_callbacks.nonlinear_fit_predict(
            train,
            labels,
            test,
            device=device,
            seed=seed,
            fit_threads=fit_threads,
        )


# Snapshot implementations once: replacing a class method must retain the
# caller's Python contract instead of silently invoking a native classifier.
_BUILTIN_CLASSIFIER_METHODS = {
    cls: {
        name: getattr(cls, name)
        for name in ("compute", "not_applicable", "compute_partitioned", "_fit_predict")
    }
    for cls in (LinearC2ST, NonLinearC2ST)
}
