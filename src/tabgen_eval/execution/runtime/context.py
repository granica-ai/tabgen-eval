"""Typed numerical capabilities over caller-assigned source rows.

Contexts own prepared storage and execute supplied comparisons. Sampling,
uncertainty estimation, and report construction belong to their callers.
Classifier-only contexts expose their common capabilities without claiming
support for tensor distances or detailed fidelity computation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias

from numpy.typing import ArrayLike

from tabgen_eval._typing import FloatArray, SeedScalar

from .details import DetailedOutcome
from .outcomes import ClassifierOutcome, MemorizationOutcome

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.xgboost.context import XGBoostContext
    from tabgen_eval.metrics import ClassifierMetric, Metric
    from tabgen_eval.schema import Schema

    NumericalContext: TypeAlias = TorchContext | XGBoostContext


class ContextMetadata(Protocol):
    """Metadata, applicability, and batching shared by numerical contexts."""

    category_mismatch: bool
    n_numeric: int
    n_categorical: int
    classifier_width: int
    workers: int

    @property
    def execution_info(self) -> dict[str, object]:
        """Return fresh report metadata describing implementation, device and precision."""
        ...

    def execution_info_for(self, metric: Metric) -> dict[str, object]:
        """Describe the implementation and routing used for one metric.

        Parameters
        ----------
        metric : Metric
            Metric whose arithmetic, storage or automatic routing should be recorded.

        Returns
        -------
        dict
            JSON-compatible execution metadata. Return an independent mapping so
            per-metric additions cannot contaminate another metric's provenance.
        """
        ...

    def batch_size(self, metric: Metric, n: int) -> int:
        """Bound how many source-index draws the estimator submits together.

        Parameters
        ----------
        metric : Metric
            Metric family and implementation-specific workspace requirements.
        n : int
            Nominal sampled rows per pool before metric-specific missing-row filters.

        Returns
        -------
        int
            Positive maximum batch length. This bounds estimator index/gather work;
            it does not imply all draws are simultaneously resident on a GPU or
            replace checks on the live workspace for one draw.
        """
        ...

    def not_applicable(self, metric: Metric, schema: Schema) -> set[str]:
        """Identify statistics unavailable for the schema or prepared reference pool.

        Parameters
        ----------
        metric : Metric
            Requested built-in statistic definitions and configuration.
        schema : Schema
            Effective schema corresponding to the prepared column order.

        Returns
        -------
        set of str
            Statistic names for estimator-level not-applicable flags. Reference-only
            checks, such as the existence of rare Conditional slices, may inspect
            prepared data but must not fit against candidate rows.
        """
        ...


class ClassifierContext(ContextMetadata, Protocol):
    """Execute classifiers with caller-supplied source folds and seeds."""

    def compute_classifier(
        self,
        metric: ClassifierMetric,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        reference_folds: ArrayLike,
        candidate_folds: ArrayLike,
        seeds: Sequence[SeedScalar],
        *,
        ideal: bool = False,
        point: bool = False,
        fit_threads: int | None = None,
    ) -> list[ClassifierOutcome]:
        """Fit and score supplied classifier folds without source-identity leakage.

        Parameters
        ----------
        metric : ClassifierMetric
            Supported classifier protocol and model/resource options.
        reference_draws, candidate_draws : array-like of int
            Source-position matrices ``(B, NR)`` and ``(B, NS)``. Preserve bootstrap
            copies; do not deduplicate before fitting or scoring.
        reference_folds, candidate_folds : array-like of int
            Fold labels with exactly the corresponding draw shapes and values in
            ``[0, metric.cv_folds)``. Every copy of a source row must retain its fold.
        seeds : iterable of int
            One exact integer in ``[0, 2**64)`` per draw. Preserve integer bits through
            native interfaces rather than converting seeds through floating point.
        ideal : bool, default=False
            Draw both classes from the reference source and require identities shared
            across classes to retain the same fold.
        point : bool, default=False
            Select point-fit work/thread limits rather than bootstrap-fit limits.
        fit_threads : int or None, default=None
            Explicit per-fit CPU budget, independent of solver selection policy.

        Returns
        -------
        list of ClassifierOutcome
            Mean held-out score and individual fold diagnostics for each draw.
            Retain nonconvergence and numerical/resource failure diagnostics so the
            estimator can decide which results enter uncertainty calculations.

        Notes
        -----
        Use each fold's training rows to fit imputation, scaling, category vocabulary
        and model/preconditioner state. Held-out rows only score the fitted model.
        A global category identity map is storage metadata, not permission to include
        held-out categories in fitted features. Fitting uses private seeded RNG state.
        """
        ...


class ExecutionContext(ClassifierContext, Protocol):
    """Full tensor/sample contract implemented by the tensor execution context."""

    def compute(
        self,
        metric: Metric,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Compute sample-based statistics for matched source-index batches.

        Parameters
        ----------
        metric : Metric
            Supported sample-based built-in, such as Shape, Trend, Conditional or PRDC.
        reference_draws, candidate_draws : array-like of int
            Two-dimensional source positions of shapes ``(B, NR)`` and ``(B, NS)``.
            Batch lengths must match; draw sizes may differ. Preserve row order and
            duplicates, and reject out-of-range/noninteger indices before gathering.
        ideal : bool, default=False
            Interpret both draw matrices against the reference source pool when true;
            otherwise candidate positions index the candidate pool.

        Returns
        -------
        dict of str to numpy.ndarray
            One host statistic array of shape ``(B,)`` per declared statistic, in
            submitted order. Noncomputable draw/statistic combinations contain NaN.

        Notes
        -----
        The estimator owns real-real pool separation, sampling and interval assembly.
        Execute only the supplied draws and retain immutable source storage. Apply
        missingness according to the metric rather than filtering all metrics alike.
        """
        ...

    def compute_detailed(
        self,
        metric: Metric,
        reference_draw: ArrayLike,
        candidate_draw: ArrayLike,
        *,
        ideal: bool = False,
    ) -> DetailedOutcome:
        """Return a fixed draw's aggregates and descriptive fidelity tables together.

        One-dimensional source positions select the same arithmetic and reduction
        as ``compute``. Only requested point comparisons collect details; reference
        and bootstrap draws do not create per-entry uncertainty or detail tables.
        """
        ...

    def compute_memorization(
        self, metric: Metric, reference_draws: ArrayLike, candidate_draws: ArrayLike
    ) -> list[MemorizationOutcome]:
        """Return one requested training-referenced metric per supplied draw."""
        ...

    def compute_memorization_group(
        self,
        metrics: Sequence[Metric],
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        holdout_draws: ArrayLike | None = None,
    ) -> list[dict[str, MemorizationOutcome]]:
        """Share geometry and searches across compatible training-referenced metrics.

        Return a list of dictionaries keyed by metric name, each containing a
        MemorizationOutcome. Draw positions preserve multiplicities and do not
        assign any resampling or uncertainty policy. Optional holdout positions
        address the independent real pool used only by DCR's baseline.
        """
        ...


class ContextProvider(Protocol):
    """Select or reuse a context for an already-resolved computation."""

    def context_for(self, metric: Metric) -> NumericalContext | None: ...


class UnsupportedExecutionError(RuntimeError):
    """An explicit implementation/device/precision combination is not available."""


def cpu_batch_size(
    metric: Metric, n: int, *, n_numeric: int, n_categorical: int, classifier_width: int
) -> int:
    """Choose a CPU replicate batch from index and conservative gather budgets.

    Parameters
    ----------
    metric : Metric
        Family determines whether to estimate expanded classifier feature width.
        Each comparison gathers two roles, including real-only comparisons that
        address one shared source pool.
    n : int
        Sampled rows per pool before missingness filtering.
    n_numeric, n_categorical : int
        Prepared source-column counts.
    classifier_width : int
        Conservative expanded width used only for classifier gather estimates.

    Returns
    -------
    int
        Between one and 64 draws, limiting index storage to roughly 8 MiB and
        estimated numeric gather storage to 256 MiB. A single oversized draw
        still returns one; algorithm-specific memory checks remain necessary.
    """
    family = metric.family.value
    pools = 2
    width = classifier_width if family == "estimator_based" else n_numeric + n_categorical
    return max(
        1,
        min(
            64,
            (8 * 1024**2) // max(8 * pools * n, 1),
            (256 * 1024**2) // max(8 * pools * n * width, 1),
        ),
    )
