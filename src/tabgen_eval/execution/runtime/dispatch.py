"""Share host preparation and choose cached numerical contexts per evaluation.

Metric evaluators ask an ``ExecutionSession`` for a metric context while
retaining ownership of sampling, source folds and report assembly. Exact native
built-ins can share typed category identities and tensor storage; custom
classes or overridden metric methods retain their Python execution contracts.

Preparation is lazy, and contexts are cached by resolved device, precision and
workspace budget only within this session. Workload routing uses capped draw
sizes and compatible resident storage, avoiding unconditional GPU uploads or
CUDA probes for small automatic workloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd

from tabgen_eval.execution.runtime.details import DetailedOutcome
from tabgen_eval.schema import Schema

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.context import ExecutionContext, NumericalContext
    from tabgen_eval.execution.runtime.memory import MemoryPlan
    from tabgen_eval.execution.runtime.routing import Workload
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.metrics.conditional import Conditional

    class _WorkspaceContext(Protocol):
        """Optional mutable workspace allowance on a prepared numerical context."""

        gpu_workspace_mb: int


from tabgen_eval.execution.runtime.options import validate_options
from tabgen_eval.preprocessing.columns import PreparedColumns

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.xgboost.context import XGBoostContext
    from tabgen_eval.preprocessing.sources import PreparedSources


def supports(metric: Metric) -> bool:
    """Check that native dispatch preserves the metric's declared public contract.

    Parameters
    ----------
    metric : Metric
        Candidate metric instance, including any instance-level method overrides.

    Returns
    -------
    bool
        True only for exact supported built-in classes with unchanged relevant
        methods, metric name and statistic names. Subclasses and customized
        instances are not silently redirected around their Python behavior.
    """
    from tabgen_eval.metrics import (
        DCR,
        PRDC,
        Concentration,
        Conditional,
        LinearC2ST,
        NearMatchRate,
        NonLinearC2ST,
        ReplicationRatio,
        Shape,
        Trend,
    )
    from tabgen_eval.metrics.c2st import _BUILTIN_CLASSIFIER_METHODS
    from tabgen_eval.metrics.concentration import concentration_specs

    cls = type(metric)
    expected_stats = cls.stats
    if cls is Concentration:
        assert isinstance(metric, Concentration)
        expected_stats = concentration_specs(metric.fixed_radii, metric.radius_quantiles)
    builtin_methods = cast(
        "Mapping[type[Metric], Mapping[str, object]]", _BUILTIN_CLASSIFIER_METHODS
    )
    return (
        cls
        in {
            Shape,
            Trend,
            Conditional,
            Concentration,
            PRDC,
            LinearC2ST,
            NonLinearC2ST,
            DCR,
            NearMatchRate,
            ReplicationRatio,
        }
        and not any(
            name in vars(metric)
            for name in (
                "compute",
                "not_applicable",
                "compute_partitioned",
                "_fit_predict",
            )
        )
        and all(
            getattr(cls, name) is implementation
            for name, implementation in builtin_methods.get(cls, {}).items()
        )
        and metric.name == cls.name
        and tuple(s.name for s in metric.stats) == tuple(s.name for s in expected_stats)
    )


def supports_details(metric: Metric) -> bool:
    """Whether an exact built-in supplies descriptive comparison tables."""
    return supports(metric) and metric.name in {
        "shape",
        "trend",
        "conditional_fidelity",
        "concentration",
    }


def compute_python_details(
    metric: Metric, reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema
) -> DetailedOutcome:
    """Run one Python comparison with descriptive score collection enabled."""
    if not supports_details(metric):
        raise ValueError(f"Detailed output is unavailable for {metric.name!r}")
    if metric.name == "concentration":
        from tabgen_eval.execution.python.concentration import concentration

        return concentration(reference, candidate, schema, metric, details=True)
    if metric.name == "conditional_fidelity":
        from tabgen_eval.execution.python.conditional import compute_conditional

        conditional = cast("Conditional", metric)
        return compute_conditional(
            reference,
            candidate,
            schema,
            num_bins=conditional.num_bins,
            min_count=conditional.min_count,
            rare_threshold=conditional.rare_threshold,
            details=True,
            requested_components=conditional.requested_components,
        )
    from tabgen_eval.execution.python.fidelity import compute_shape, compute_trend

    function = compute_shape if metric.name == "shape" else compute_trend
    return function(reference, candidate, schema, details=True)


def supports_prepared_evaluation(metrics: Sequence[Metric], schema: Schema) -> bool:
    """Decide whether all consumers permit reusing prepared source storage.

    Parameters
    ----------
    metrics : sequence of Metric
        Nonempty set of requested metrics. Each must use an unmodified tensor
        built-in/repeated-score/input-validation contract.
    schema : Schema
        Schema used by metric execution selection.

    Returns
    -------
    bool
        Whether fused prepared evaluation is eligible and Torch is importable.
        False keeps original DataFrames for Python/custom consumers.

    Notes
    -----
    This is a conservative capability gate without preparing values or invoking kernels.
    Preparation can decline when Torch is absent; built-in execution still
    requires Torch. Failures inside an installed Torch propagate.
    Custom validation must see the ordinary evaluation path.
    """
    from tabgen_eval.metrics.base import Metric

    if not metrics or not all(
        supports(metric)
        and metric.bootstrapped
        and type(metric).validate_inputs is Metric.validate_inputs
        and "validate_inputs" not in vars(metric)
        for metric in metrics
    ):
        return False
    from tabgen_eval.execution.torch.tensors import torch_module

    return torch_module(required=False) is not None


class ExecutionSession:
    """Cache prepared column identities and tensor contexts for one evaluation.

    Parameters
    ----------
    reference, candidate : pandas.DataFrame
        Validated source pools unless compatible prepared columns are
        supplied. They remain read-only from the session's perspective.
    schema : Schema
        Effective feature order shared by all cached contexts.
    n_jobs : int, default=1
        Joblib-style CPU worker budget passed to each newly created context.
    n_eff : int or None, default=None
        Nominal sampled rows after size caps; defaults to the smaller source pool.
        Used only for resource/crossover routing, not source-row truncation.
    n_bootstrap : int, default=0
        Requested replicate count used by crossover routing.
    columns : PreparedColumns or None, default=None
        Optional already prepared owned host pools, including the fused Arrow
        preparation result. Otherwise column preparation occurs on first access.
    retain_labels : bool, default=False
        Keep existing categorical dictionaries when preparing descriptive details.

    Notes
    -----
    Cache keys contain implementation, resolved device, numeric dtype and GPU workspace
    budget. Contexts contain numerical pools/executors, not fitted fold models.
    The cache is evaluation-local and not a global or thread-safe storage service.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        n_jobs: int = 1,
        n_eff: int | None = None,
        n_bootstrap: int = 0,
        columns: PreparedColumns | None = None,
        retain_labels: bool = False,
        classifier_preprocessing: bool = False,
    ) -> None:
        """Retain source references/options without preparing arrays or probing CUDA."""
        self.reference, self.candidate, self.schema = reference, candidate, schema
        self.n_jobs = n_jobs
        self.n_eff = min(len(reference), len(candidate)) if n_eff is None else n_eff
        self.n_bootstrap = n_bootstrap
        self._columns: PreparedColumns | None = columns
        self.retain_labels = retain_labels or classifier_preprocessing
        self.classifier_preprocessing = classifier_preprocessing
        self._contexts: dict[tuple[str, str, str, int], TorchContext | XGBoostContext] = {}
        self._classifier_sources: dict[tuple[str, str], PreparedSources] = {}
        from tabgen_eval.execution.runtime.memory import ResourcePlanner

        self._memory = ResourcePlanner()

    @staticmethod
    def _budget_key(metric: Metric) -> int:
        """Share automatically managed pools across changing live allowances."""
        from tabgen_eval.execution.runtime.memory import automatic

        return 0 if automatic(metric, "gpu_workspace_mb") else metric.gpu_workspace_mb

    def _plan_memory(
        self,
        metric: Metric,
        device: str,
        *,
        new_source_bytes: int = 0,
        persistent_input_bytes: int = 0,
    ) -> MemoryPlan:
        return self._memory.resolve(
            metric,
            device,
            new_source_bytes=new_source_bytes,
            persistent_input_bytes=persistent_input_bytes,
        )

    def _refresh_context_memory(
        self, metric: Metric, context: NumericalContext
    ) -> NumericalContext:
        """Refresh after source placement; resident pools are already charged by free RAM."""
        info = context.execution_info
        self._plan_memory(
            metric,
            cast("str", info["device"]),
            persistent_input_bytes=cast("int", info.get("persistent_input_bytes", 0)),
        )
        if hasattr(context, "gpu_workspace_mb"):
            cast("_WorkspaceContext", context).gpu_workspace_mb = metric.gpu_workspace_mb
        context._automatically_managed_memory = True
        return context

    @property
    def columns(self) -> PreparedColumns:
        """Lazily prepare and retain one shared host encoding for this evaluation."""
        if self._columns is None:
            options = {"retain_labels": True} if self.retain_labels else {}
            self._columns = (
                PreparedColumns.prepare_single(self.reference, self.schema, **options)
                if self.reference is self.candidate
                else PreparedColumns.prepare(self.reference, self.candidate, self.schema, **options)
            )
        return self._columns

    def classifier_sources(self, metric: Metric, device: str) -> PreparedSources:
        """Share matched source tensors across both classifier implementations."""
        from tabgen_eval.preprocessing import prepare_sources
        from tabgen_eval.preprocessing.sources import ensure_category_labels

        key = (device, metric.dtype)
        if key not in self._classifier_sources:
            self._columns = ensure_category_labels(self.columns, self.reference, self.candidate)
            self._classifier_sources[key] = prepare_sources(
                self.reference,
                self.candidate,
                self.schema,
                device=device,
                dtype=metric.dtype,
                columns=self.columns,
            )
        return self._classifier_sources[key]

    def context_for(self, metric: Metric) -> TorchContext | XGBoostContext | None:
        """Select a native context, or retain a custom metric's computation.

        Parameters
        ----------
        metric : Metric
            Built-in/configuration to route using this evaluation's effective workload.

        Returns
        -------
        TorchContext or XGBoostContext or None
            Context implementing the selected metric's numerical operations, or
            ``None`` when the metric provides a custom computation.

        Raises
        ------
        ValueError
            A supported metric carries malformed execution options.
        ImportError
            An explicitly required numerical dependency is absent or broken.
        UnsupportedExecutionError
            Requested device/precision/framework combination has no implementation.

        Notes
        -----
        An automatic context records the selection reason under the metric name.
        Tensor pools are uploaded/copied only after selection, and matching contexts
        are reused across metrics. Explicit precision/device requests do not silently
        fall back to a missing numerical dependency or a different precision.
        """
        self._plan_memory(metric, "cpu")
        if not supports(metric):
            return None
        validate_options(metric.device, metric.dtype, metric.gpu_workspace_mb)
        classifier = metric.name in {"linear_c2st", "nonlinear_c2st"}
        if classifier and self._columns is None:
            self.retain_labels = True
        if metric.name == "nonlinear_c2st":
            from tabgen_eval.execution.xgboost.context import XGBoostContext
            from tabgen_eval.execution.xgboost.device import select_device

            workload = self._workload(metric)
            device = select_device(
                metric,
                rows=2 * workload.n_eff,
                numeric=workload.numeric,
                cardinalities=workload.cardinalities,
                new_source_bytes=(
                    0
                    if workload.cuda_resident
                    or (metric.device, metric.dtype) in self._classifier_sources
                    else workload.persistent_bytes(metric.dtype)
                ),
            )
            key = ("xgboost", device, metric.dtype, self._budget_key(metric))
            if key not in self._contexts:
                self._plan_memory(
                    metric,
                    device,
                    new_source_bytes=(
                        0
                        if (device, metric.dtype) in self._classifier_sources
                        else workload.persistent_bytes(metric.dtype)
                    ),
                )
                sources = self.classifier_sources(metric, device)
                self._contexts[key] = XGBoostContext(
                    sources,
                    n_jobs=self.n_jobs,
                    gpu_workspace_mb=metric.gpu_workspace_mb,
                )
            return self._refresh_context_memory(metric, self._contexts[key])
        from tabgen_eval.execution.torch.context import TorchContext
        from tabgen_eval.execution.torch.tensors import torch_module

        torch_module(required=True)
        device, reason = TorchContext.select_device(metric, self._workload(metric))
        from tabgen_eval.preprocessing.sources import matching_is_noop

        share = self.classifier_preprocessing and matching_is_noop(
            self.reference,
            self.candidate,
            self.schema.numeric,
            metric.dtype,
        )
        key = (
            "torch_classifier" if classifier and not share else "torch",
            device,
            metric.dtype,
            self._budget_key(metric),
        )
        if key not in self._contexts:
            self._plan_memory(
                metric,
                device,
                new_source_bytes=(
                    0
                    if (classifier or share) and (device, metric.dtype) in self._classifier_sources
                    else self._workload(metric).persistent_bytes(metric.dtype)
                ),
            )
            if classifier or share:
                sources = self.classifier_sources(metric, device)
                context = TorchContext(
                    sources.tensors, n_jobs=self.n_jobs, gpu_workspace_mb=metric.gpu_workspace_mb
                )
                context.reference_cardinalities = sources.reference_cardinalities
                context.category_values = sources.category_values
                context.preprocessing_precision = sources.precision
                context.source_preparation_seconds = sources.preparation_seconds
                self._contexts[key] = context
            else:
                self._contexts[key] = TorchContext.from_columns(
                    self.columns,
                    device=device,
                    dtype=metric.dtype,
                    n_jobs=self.n_jobs,
                    gpu_workspace_mb=metric.gpu_workspace_mb,
                )
        # This key is created only by the tensor branch above; classifier-only
        # contexts use separate cache prefixes and return before this point.
        tensor_context = cast("TorchContext", self._contexts[key])
        if metric.device == "auto":
            tensor_context.routes[metric.name] = reason
        return self._refresh_context_memory(metric, tensor_context)

    def _workload(self, metric: Metric) -> Workload:
        """Estimate a metric's capped workload and genuinely reusable CUDA storage.

        Parameters
        ----------
        metric : Metric
            Precision/workspace configuration whose resident context may be reused.

        Returns
        -------
        Workload
            Column/domain/source sizes, estimated complete rows and bootstrap count.
            Storage counts as resident only on the caller's current CUDA device with
            matching precision and budget. Existing compatible CUDA contexts imply
            initialization; otherwise no current-device probe is performed here.
        """
        from tabgen_eval.execution.runtime.routing import Workload

        columns = self.columns
        from tabgen_eval.preprocessing.sources import matching_is_noop

        classifier = metric.name in {"linear_c2st", "nonlinear_c2st"}
        reuse_classifier = classifier or (
            self.classifier_preprocessing
            and matching_is_noop(self.reference, self.candidate, self.schema.numeric, metric.dtype)
        )
        matches = [
            key
            for key in self._contexts
            if not classifier
            and key[0] == "torch"
            and key[1].startswith("cuda")
            and key[2:] == (metric.dtype, self._budget_key(metric))
        ]
        source_devices = [
            device
            for device, dtype in self._classifier_sources
            if reuse_classifier and device.startswith("cuda") and dtype == metric.dtype
        ]
        resident = False
        if matches or source_devices:
            from tabgen_eval.execution.torch.tensors import torch_module

            # Cached storage on another GPU, or in another precision/budget,
            # cannot be reused by this context. CUDA is already initialized
            # whenever there is an existing device context to inspect.
            current = f"cuda:{torch_module().cuda.current_device()}"
            resident = any(key[1] == current for key in matches) or current in source_devices
        complete = min(
            int(self.n_eff * table.complete.mean()) if table.n_rows else 0
            for table in (columns.reference, columns.candidate)
        )
        distance_pools = None
        if metric.name in {"dcr", "near_match_rate", "replication_ratio"}:
            distance_pools = (columns.reference, columns.candidate)
            if metric.name == "dcr" and columns.holdout is not None:
                distance_pools += (columns.holdout,)
        return Workload(
            self.n_eff,
            columns.n_numeric,
            columns.n_categorical,
            columns.cardinalities,
            columns.reference.n_rows,
            columns.candidate.n_rows,
            complete,
            self.n_bootstrap,
            resident,
            shared_source=columns.reference is columns.candidate,
            holdout_rows=0 if columns.holdout is None else columns.holdout.n_rows,
            draw_rows=None if distance_pools is None else tuple(p.n_rows for p in distance_pools),
            usable_rows=None
            if distance_pools is None
            else tuple(int(p.complete.sum()) for p in distance_pools),
        )


def context_for(
    metric: Metric,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    n_jobs: int = 1,
) -> TorchContext | XGBoostContext | None:
    """Create a one-off execution session and select one metric context.

    Parameters
    ----------
    metric : Metric
        Metric whose device and precision options determine selection.
    reference, candidate : pandas.DataFrame
        Validated source pools in original source order.
    schema : Schema
        Effective numeric/categorical feature declarations.
    n_jobs : int, default=1
        CPU worker budget; does not create concurrent CUDA streams.

    Returns
    -------
    TorchContext or XGBoostContext or None
        Context implementing the selected metric's numerical operations, or a
        custom-metric sentinel. Repeated standalone calls do not share preparation;
        evaluation-wide callers should use one session.
    """
    return ExecutionSession(reference, candidate, schema, n_jobs=n_jobs).context_for(metric)


def compute_metric(
    metric: Metric, reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema
) -> dict[str, float] | None:
    """Evaluate one full-pool sample metric through numerical dispatch.

    Parameters
    ----------
    metric : Metric
        Supported sample-based built-in with ordinary execution options.
    reference, candidate : pandas.DataFrame
        Validated source pools; all source positions form one draw each.
    schema : Schema
        Effective feature declarations corresponding to the source pools.

    Returns
    -------
    dict of str to float or None
        Scalar statistic values for the single full-pool draw, or ``None`` when
        native dispatch declines and the metric should run its Python path.

    Notes
    -----
    No repeated sampling, real-real comparison, or classifier fold construction is
    performed here. Those protocols use the corresponding context methods.
    """
    import numpy as np

    context = context_for(metric, reference, candidate, schema)
    if context is None:
        return None
    values = cast("ExecutionContext", context).compute(
        metric, np.arange(len(reference))[None, :], np.arange(len(candidate))[None, :]
    )
    return {name: float(value[0]) for name, value in values.items()}
