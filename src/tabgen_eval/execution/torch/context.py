"""Coordinate compiled executors over one pair of owned tensor source pools.

Evaluation sessions select a device, transfer shared columns once and reuse this
context across supported metrics with matching precision/workspace settings.
CPU metrics share an ATen/C++ executor; CUDA distance, fidelity, Conditional and
classifier executors are loaded lazily by family. The context delegates numerical
work while evaluation and uncertainty orchestration own row draws and source-fold
assignment. Report modules assemble the returned numerical outcomes.

Automatic routing combines host workload estimates with minimum framework
versions, current-device free memory and conservative temporary-buffer estimates.
Explicit requests retain their requested device/precision or raise. Persistent
source pools are separate from draw-local GPU workspace limits.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, TypeAlias
from weakref import proxy

from joblib import effective_n_jobs
from numpy.typing import ArrayLike

from tabgen_eval._typing import FloatArray, Precision, SeedScalar
from tabgen_eval.execution.runtime.context import UnsupportedExecutionError
from tabgen_eval.execution.runtime.details import DetailedOutcome
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome, MemorizationOutcome
from tabgen_eval.execution.torch.tensors import resolve_device, torch_module

if TYPE_CHECKING:
    from torch import Tensor

    from tabgen_eval.execution.runtime.routing import Workload
    from tabgen_eval.execution.torch.concentration import ConcentrationExecutor
    from tabgen_eval.execution.torch.cpu.executor import CppExecutor
    from tabgen_eval.execution.torch.cuda.classifier.executor import ClassifierExecutor
    from tabgen_eval.execution.torch.distance.executor import DistanceExecutor
    from tabgen_eval.execution.torch.fidelity.conditional import ConditionalExecutor
    from tabgen_eval.execution.torch.fidelity.executor import FidelityExecutor
    from tabgen_eval.execution.torch.tensors import TensorInputs, TensorTable
    from tabgen_eval.metrics import ClassifierMetric, Metric
    from tabgen_eval.preprocessing.columns import PreparedColumns
    from tabgen_eval.preprocessing.sources import ObservationPrecision
    from tabgen_eval.schema import Schema

    TensorExecutor: TypeAlias = (
        CppExecutor
        | DistanceExecutor
        | FidelityExecutor
        | ConditionalExecutor
        | ClassifierExecutor
        | ConcentrationExecutor
    )


def cuda_supported(torch: ModuleType) -> bool:
    """Check independent Torch/CUDA/Triton minima without launching kernels.

    Parameters
    ----------
    torch : module
        Already imported Torch package whose version and CUDA build are inspected.

    Returns
    -------
    bool
        Whether Torch >=2.6, CUDA >=11.8 and Triton >=3.2 are installed.
        Versions need not belong to a predefined combination. A missing Triton returns
        false; hardware/driver availability is checked separately by device selection.
    """
    try:
        triton = version("triton")
    except PackageNotFoundError:
        return False
    for installed, minimum in (
        (torch.__version__, (2, 6)),
        (torch.version.cuda, (11, 8)),
        (triton, (3, 2)),
    ):
        parsed = re.match(r"^(\d+)\.(\d+)(?:\.|\+|$)", str(installed))
        if parsed is None or tuple(map(int, parsed.groups())) < minimum:
            return False
    return True


def triton_installed() -> bool:
    """Check discoverability without importing Triton or hiding broken parents.

    Returns
    -------
    bool
        Whether ``find_spec`` can discover the top-level Triton package. Only an
        actual missing Triton is treated as unavailable; unrelated import errors
        propagate. This is not a version or kernel-launch compatibility check.
    """
    try:
        return find_spec("triton") is not None
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        return False


class TorchContext:
    """Own prepared tensor pools and lazily create metric executors.

    Parameters
    ----------
    inputs : TensorInputs
        Immutable-by-contract tables on a single CPU/CUDA device in a selected
        floating precision. Contexts retain ownership for one evaluation only.
    n_jobs : int, default 1
        Joblib-style CPU worker budget. GPU draws execute sequentially on the
        caller's current stream; this setting does not create CUDA workers.
    gpu_workspace_mb : int, default 2048
        Maximum temporary metric workspace in MiB, excluding persistent input
        pools. Each executor checks draw-local preprocessing and scratch.

    Notes
    -----
    Construction does not load compiled metric operators. CUDA construction
    activates a compatible bundled Triton host cache if one is installed.
    Executors disable autocast locally without changing global precision,
    thread, device or stream settings. Cached executors hold weak context proxies,
    so the owner and its inputs can be released without waiting for cyclic GC.
    The context must outlive calls through these internal cached adapters.
    """

    preprocessing_precision: ObservationPrecision
    source_preparation_seconds: float
    _automatically_managed_memory: bool

    def __init__(
        self, inputs: TensorInputs, *, n_jobs: int = 1, gpu_workspace_mb: int = 2048
    ) -> None:
        """Retain validated tensor pools, resolve workers and initialize lazy caches.

        No metric executors are created here. CUDA contexts may seed a compatible
        bundled Triton host cache; this does not gather rows or fit metric geometry.
        Reject a zero CPU worker request before executors can use its budget.
        """
        from tabgen_eval.execution.runtime.options import validate_options

        validate_options(str(inputs.device), inputs.dtype, gpu_workspace_mb)
        self.inputs = inputs
        self.reference, self.candidate = inputs.reference, inputs.candidate
        self.holdout = inputs.holdout
        self.category_mismatch = inputs.category_mismatch
        self.reference_cardinalities: tuple[int, ...] | None = None
        self.category_values: tuple[tuple[object, ...], ...] | None = None
        self.n_numeric, self.n_categorical = (
            len(inputs.numeric_names),
            len(inputs.categorical_names),
        )
        self.classifier_width = self.n_numeric + sum(inputs.cardinalities) + self.n_categorical
        if n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        self.workers: int = effective_n_jobs(n_jobs)
        self.gpu_workspace_mb = gpu_workspace_mb
        self._cpu: CppExecutor | None = None
        self._distances: DistanceExecutor | None = None
        self._fidelity: FidelityExecutor | None = None
        self._conditional: ConditionalExecutor | None = None
        self._classifier: ClassifierExecutor | None = None
        self._concentration: ConcentrationExecutor | None = None
        self.routes: dict[str, str] = {}
        if inputs.device.type == "cuda":
            from tabgen_eval.execution.torch.cuda.triton.host import install

            install()

    @classmethod
    def from_columns(
        cls,
        columns: PreparedColumns,
        *,
        device: str = "cpu",
        dtype: Precision = "float64",
        n_jobs: int = 1,
        gpu_workspace_mb: int = 2048,
    ) -> TorchContext:
        """Transfer prepared host pools and retain reference-domain size metadata.

        Parameters
        ----------
        columns : PreparedColumns
            Shared host encoding from pandas or fused Arrow preparation.
        device : str, default="cpu"
            Already selected destination; direct ``auto`` callers use storage
            availability resolution rather than metric workload selection.
        dtype : {"float32", "float64"}, default="float64"
            Precision for both copied numeric source matrices.
        n_jobs : int, default=1
            Joblib-style CPU worker budget.
        gpu_workspace_mb : int, default=2048
            Temporary GPU workspace in MiB, excluding persistent source pools.

        Returns
        -------
        TorchContext
            Context owning independent row-major tensors and empty executor caches.
            Reference-only category counts remain available for downstream sizing.
        """
        context = cls(
            columns.to_tensors(device=device, dtype=dtype),
            n_jobs=n_jobs,
            gpu_workspace_mb=gpu_workspace_mb,
        )
        context.reference_cardinalities = columns.reference_cardinalities
        context.category_values = columns.category_values
        return context

    @staticmethod
    def has_kernel(metric: Metric, *, device: str, dtype: Precision) -> bool:
        """Check declared metric/device-kind/precision support without loading kernels.

        Parameters
        ----------
        metric : Metric
            Exact supported built-in; customized classes/contracts are rejected.
        device : {"cpu", "cuda"}
            Device kind, not an indexed CUDA device string.
        dtype : {"float32", "float64"}
            Requested numerical precision.

        Returns
        -------
        bool
            Whether a kernel contract exists. CPU and CUDA Linear C2ST support
            FP32 and FP64. This does not check installed libraries,
            framework versions, device availability or live workspace.
        """
        from tabgen_eval.execution.runtime.dispatch import supports

        if not supports(metric) or dtype not in {"float32", "float64"}:
            return False
        if device == "cpu":
            return True
        return device == "cuda" and metric.name in {
            "shape",
            "prdc",
            "dcr",
            "near_match_rate",
            "replication_ratio",
            "concentration",
            "trend",
            "conditional_fidelity",
            "linear_c2st",
        }

    @classmethod
    def select_device(cls, metric: Metric, workload: Workload | None = None) -> tuple[str, str]:
        """Resolve numerical placement and record the reason for the decision.

        Parameters
        ----------
        metric : Metric
            Exact supported built-in carrying device, precision and workspace/model
            options. Explicit devices bypass automatic workload crossover thresholds.
        workload : Workload or None, default=None
            Capped draw/schema sizes and compatible cached-storage information. With
            no estimate, automatic selection considers available CUDA directly.

        Returns
        -------
        tuple of str and str
            Canonical ``cpu`` or ``cuda:index`` plus an auditable routing reason.
            Unindexed CUDA retains the caller's current device.

        Raises
        ------
        UnsupportedExecutionError
            Requested precision/device has no supported kernel or the selected CUDA
            framework matrix is unsupported. Automatic CPU fallback must still
            support the requested precision; it never silently changes dtype.
        ImportError
            Explicit CUDA requires an absent/broken kernel dependency or library.
        RuntimeError
            Explicit CUDA is unavailable, or native/kernel initialization fails.
        ValueError
            A requested device/index is malformed or unavailable.

        Notes
        -----
        Small or known oversized automatic workloads remain on CPU without CUDA
        initialization. Candidate CUDA workloads check current-device free memory,
        including persistent pools when not reusable and a safety reserve. These
        estimates do not replace live executor workspace checks.

        A selected CUDA path loads/initializes its kernel provider before returning;
        source builds may compile if no compatible packaged library/cache exists.
        The method preserves the caller's current device, stream and precision state.
        """
        torch = torch_module()
        from tabgen_eval.execution.runtime.memory import automatic, plan_metric_memory

        reason = "explicit_device"
        if metric.device == "auto":
            worthwhile, reason = (
                workload.cuda_candidate(metric) if workload else (True, "available_cuda_kernel")
            )
            if worthwhile and workload:
                minimum = workload.minimum_workspace(metric)
                if (
                    not automatic(metric, "gpu_workspace_mb")
                    and minimum > metric.gpu_workspace_mb * 1024**2
                ):
                    worthwhile, reason = False, "gpu_workspace_too_small"
                elif (
                    metric.name == "linear_c2st"
                    and not automatic(metric, "memory_limit_mb")
                    and minimum - 16 * 1024**2 > metric.memory_limit_mb * 1024**2
                ):
                    worthwhile, reason = False, "model_workspace_too_small"
            use_cuda = (
                worthwhile
                and cls.has_kernel(metric, device="cuda", dtype=metric.dtype)
                and torch.cuda.is_available()
                and triton_installed()
            )
            if worthwhile and not use_cuda:
                reason = "cuda_or_triton_unavailable"
            if use_cuda and not cuda_supported(torch):
                use_cuda, reason = False, "cuda_version_below_minimum"
            if use_cuda and metric.name == "linear_c2st":
                from tabgen_eval.execution.torch.cuda.extension import available

                use_cuda = available()
                if not use_cuda:
                    reason = "cuda_classifier_library_unavailable"
            if use_cuda and workload:
                # Respect the caller's current CUDA device; do not change it or
                # select another process's GPU. Inputs are persistent, separate
                # from the temporary workspace budget.
                new_sources = (
                    0 if workload.cuda_resident else workload.persistent_bytes(metric.dtype)
                )
                try:
                    plan = plan_metric_memory(metric, "cuda", new_source_bytes=new_sources)
                except MemoryError:
                    use_cuda, reason = False, "insufficient_free_cuda_memory"
                    plan = None
                required = workload.minimum_workspace(metric)
                if use_cuda and required > metric.gpu_workspace_mb * 1024**2:
                    use_cuda, reason = False, "gpu_workspace_too_small"
                elif (
                    use_cuda
                    and metric.name == "linear_c2st"
                    and required - 16 * 1024**2 > metric.memory_limit_mb * 1024**2
                ):
                    use_cuda, reason = False, "model_workspace_too_small"
                if use_cuda and required + new_sources + 64 * 1024**2 > (
                    plan.available_device_bytes + plan.reusable_torch_cache_bytes
                ):
                    use_cuda, reason = False, "insufficient_free_cuda_memory"
            device = "cuda" if use_cuda else "cpu"
        else:
            device = metric.device
        resolved = resolve_device(device)
        if not cls.has_kernel(metric, device=resolved.type, dtype=metric.dtype):
            raise UnsupportedExecutionError(
                f"No torch metric implementation for {metric.name!r} on {resolved} "
                f"with {metric.dtype}. Shape/PRDC/DCR/Trend/Conditional support CPU/CUDA FP32 "
                "and FP64; Linear C2ST supports CPU/CUDA FP32 and FP64."
            )
        if resolved.type == "cuda":
            if not triton_installed():
                raise ImportError("CUDA execution requires Triton; install 'tabgen-eval[cuda]'.")
            if not cuda_supported(torch):
                raise UnsupportedExecutionError(
                    "CUDA execution requires Torch >=2.6, CUDA >=11.8 and Triton >=3.2; "
                    "use CPU or install dependencies meeting these minimum versions."
                )
            plan_metric_memory(
                metric,
                str(resolved),
                new_source_bytes=(
                    workload.persistent_bytes(metric.dtype)
                    if workload is not None and not workload.cuda_resident
                    else 0
                ),
            )
            if metric.name == "linear_c2st":
                from tabgen_eval.execution.torch.cuda.extension import library as cuda_kernels
            elif metric.name in {"shape", "conditional_fidelity"}:
                from tabgen_eval.execution.torch.fidelity.grouped_ops import kernels as cuda_kernels
            elif metric.name == "concentration":
                # Concentration loads its Triton kernels through its own executor.
                return str(resolved), reason
            elif metric.name == "trend":
                from tabgen_eval.execution.torch.fidelity.ops import cuda_kernels
            else:
                from tabgen_eval.execution.torch.distance.ops import cuda_kernels

            cuda_kernels()
        else:
            plan_metric_memory(metric, "cpu")
        return str(resolved), reason

    @classmethod
    def execution_device(cls, metric: Metric) -> str:
        """Return placement without capped-workload information or session pool reuse."""
        return cls.select_device(metric)[0]

    @property
    def execution_info(self) -> dict[str, object]:
        """Return fresh implementation/device/dtype metadata for report provenance."""
        return {
            "implementation": "triton" if self.inputs.device.type == "cuda" else "aten_cpp",
            "device": str(self.inputs.device),
            "dtype": self.inputs.dtype,
            "source_pools": (1 if self.reference is self.candidate else 2)
            + int(self.holdout is not None),
            "persistent_input_bytes": self.inputs.nbytes,
        }

    def execution_info_for(self, metric: Metric) -> dict[str, object]:
        """Add a metric's arithmetic, implementation and recorded routing metadata.

        Parameters
        ----------
        metric : Metric
            Requested metric whose family-specific arithmetic should be reported.

        Returns
        -------
        dict
            Independent metadata mapping. CUDA classifier execution identifies its
            native cuBLAS/cuSOLVER host library; other families report distance,
            correlation or contingency policies. Automatic reasons are those most
            recently recorded for this metric name on the shared context.
        """
        info = self.execution_info
        if hasattr(metric, "_memory_plan"):
            info["memory_plan"] = dict(metric._memory_plan)
        if metric.name == "linear_c2st" and hasattr(self, "preprocessing_precision"):
            info["preprocessing"] = dict(self.preprocessing_precision)
            info["source_preparation_seconds"] = self.source_preparation_seconds
        if self.inputs.device.type == "cuda" and metric.name == "linear_c2st":
            info["implementation"] = "cuda_cpp"
            info["linear_algebra"] = "cublas_cusolver"
            info["floating_arithmetic"] = "ieee"
        if metric.name in {"prdc", "dcr", "near_match_rate", "replication_ratio", "concentration"}:
            info["distance_arithmetic"] = "direct"
        elif metric.name in {"trend", "conditional_fidelity"}:
            info["correlation_arithmetic"] = "ieee"
            info["contingency_storage"] = "bounded_dense_or_occupied"
        if self._distances is not None:
            search = getattr(self._distances, "memorization_execution", {}).get(metric.name)
            if search is not None:
                info.update(search)
        if metric.name in self.routes:
            info["device_selection"] = "auto"
            info["routing_reason"] = self.routes[metric.name]
        return info

    def _refresh_metric_memory(self, metric: Metric) -> None:
        """Refresh draw admission after other contexts have acquired resident storage."""
        from tabgen_eval.execution.runtime.memory import plan_metric_memory

        plan_metric_memory(
            metric, str(self.inputs.device), persistent_input_bytes=self.inputs.nbytes
        )
        if getattr(self, "_automatically_managed_memory", False):
            self.gpu_workspace_mb = metric.gpu_workspace_mb

    def _executor(self, metric: Metric) -> TensorExecutor:
        """Lazily create and cache the implementation owning a metric's numerical work.

        Parameters
        ----------
        metric : Metric
            Exact built-in supported on this context's device and precision.

        Returns
        -------
        object
            Shared CPU executor or the cached CUDA distance/fidelity/Conditional/
            classifier executor. Constructing an executor may load compiled operators
            and prepare family-level metadata but does not choose new source draws.

        Raises
        ------
        UnsupportedExecutionError
            No implementation exists for this metric/device/precision combination.
        """
        if not self.has_kernel(metric, device=self.inputs.device.type, dtype=self.inputs.dtype):
            raise UnsupportedExecutionError(
                f"No torch metric implementation for {metric.name!r} on {self.inputs.device} "
                f"with {self.inputs.dtype}"
            )
        if metric.name == "concentration":
            if self._concentration is None:
                from tabgen_eval.execution.torch.concentration import ConcentrationExecutor

                self._concentration = ConcentrationExecutor(proxy(self))
            return self._concentration
        if self.inputs.device.type == "cpu":
            if self._cpu is None:
                from tabgen_eval.execution.torch.cpu.executor import CppExecutor

                self._cpu = CppExecutor(proxy(self))
            return self._cpu
        if metric.name in {"prdc", "dcr", "near_match_rate", "replication_ratio"}:
            if self._distances is None:
                from tabgen_eval.execution.torch.distance.executor import DistanceExecutor

                self._distances = DistanceExecutor(proxy(self))
            return self._distances
        if metric.name in {"shape", "trend"}:
            if self._fidelity is None:
                from tabgen_eval.execution.torch.fidelity.executor import FidelityExecutor

                self._fidelity = FidelityExecutor(proxy(self))
            return self._fidelity
        if metric.name == "conditional_fidelity":
            if self._conditional is None:
                from tabgen_eval.execution.torch.fidelity.conditional import ConditionalExecutor

                self._conditional = ConditionalExecutor(proxy(self))
            return self._conditional
        if metric.name == "linear_c2st":
            if self._classifier is None:
                from tabgen_eval.execution.torch.cuda.classifier.executor import ClassifierExecutor

                self._classifier = ClassifierExecutor(proxy(self))
            return self._classifier
        raise UnsupportedExecutionError(f"No CUDA executor for {metric.name!r}")

    def batch_size(self, metric: Metric, n: int) -> int:
        """Ask the selected executor for a positive estimator draw-batch bound.

        Parameters
        ----------
        metric : Metric
            Family/configuration whose index and scratch requirements determine size.
        n : int
            Nominal sampled rows per source pool before missingness filtering.

        Returns
        -------
        int
            Number of draws the estimator may submit together. CUDA execution may
            still process those draws serially to bound live device workspace.
        """
        return self._executor(metric).batch_size(metric, n)

    def gather(self, positions: ArrayLike | Tensor, *, candidate: bool = False) -> TensorTable:
        """Copy one indexed source draw after a conservative CUDA workspace check.

        Parameters
        ----------
        positions : array-like or torch.Tensor
            One-dimensional integer source positions; order and duplicates survive.
        candidate : bool, default=False
            Select candidate storage instead of reference storage.

        Returns
        -------
        TensorTable
            Fresh row-major tensors with gathered complete-row flags. Missing rows
            remain present until the consuming metric applies its missingness policy.

        Raises
        ------
        MemoryError
            Estimated draw tensor bytes plus int64 indices exceed GPU workspace.
        ValueError
            Source positions have invalid dimensions/dtypes or are out of bounds.

        Notes
        -----
        The check covers this gathered draw only, excluding persistent source pools.
        Executors remain responsible for aggregate draw/preprocessing/scratch bounds.
        """
        table = self.candidate if candidate else self.reference
        if self.inputs.device.type == "cuda":
            row_bytes = (
                table.numeric.shape[1] * table.numeric.element_size()
                + table.categorical.shape[1] * 8
                + 1
            )
            if len(positions) * (row_bytes + 8) > self.gpu_workspace_mb * 1024**2:
                raise MemoryError("one gathered draw exceeds the resolved workspace allowance")
        return table.gather(positions)

    def not_applicable(self, metric: Metric, schema: Schema) -> set[str]:
        """Delegate unavailable-statistic discovery to the metric's executor.

        Parameters
        ----------
        metric : Metric
            Requested built-in and its reference-dependent slice options.
        schema : Schema
            Effective feature schema corresponding to the context's tensor columns.

        Returns
        -------
        set of str
            Statistic names for not-applicable flags, including reference-only rare
            slice checks where needed. Candidate rows must not define applicability.
        """
        return self._executor(metric).not_applicable(metric, schema)

    def compute(
        self,
        metric: Metric,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Delegate sample-based draws using the ExecutionContext sampling contract.

        Parameters
        ----------
        metric : Metric
            Supported sample-based built-in.
        reference_draws, candidate_draws : array-like of int
            Position matrices ``(B, NR)``/``(B, NS)`` with equal batch lengths.
        ideal : bool, default=False
            Interpret both draw matrices against the reference source when true.

        Returns
        -------
        dict of str to numpy.ndarray
            Host statistic arrays of length ``B`` in draw order, retaining NaN for
            noncomputable values. The executor preserves duplicates, validates
            indices and enforces metric-specific missingness/workspace rules.
        """
        self._refresh_metric_memory(metric)
        return self._executor(metric).compute(metric, reference_draws, candidate_draws, ideal=ideal)

    def compute_detailed(
        self,
        metric: Metric,
        reference_draw: ArrayLike,
        candidate_draw: ArrayLike,
        *,
        ideal: bool = False,
    ) -> DetailedOutcome:
        """Collect one draw's fidelity breakdown alongside its original scalar reduction.

        Draws are one-dimensional source-position arrays. Detail tensors are copied
        to host by the numerical executor; typed label dictionaries stay on host.
        No additional draw, metric evaluation or uncertainty computation is made.
        Categorical Conditional details require host columns prepared with
        ``retain_labels=True`` so slice IDs can be resolved without rescanning data.
        """
        from tabgen_eval.execution.runtime.dispatch import supports_details

        if not supports_details(metric):
            raise ValueError(f"Detailed output is unavailable for {metric.name!r}")
        self._refresh_metric_memory(metric)
        if (
            metric.name == "conditional_fidelity"
            and self.n_categorical
            and self.category_values is None
        ):
            raise ValueError(
                "Conditional slice details require retained category labels; prepare "
                "columns with retain_labels=True or use evaluate(..., details=True)."
            )
        outcome = self._executor(metric).compute_detailed(
            metric, reference_draw, candidate_draw, ideal=ideal
        )
        outcome.details.metadata["column_names"] = (
            self.inputs.numeric_names + self.inputs.categorical_names
        )
        if metric.name == "conditional_fidelity" and self.category_values is not None:
            outcome.details.metadata["category_values"] = {
                self.n_numeric + i: values for i, values in enumerate(self.category_values)
            }
        return outcome

    def compute_memorization(
        self, metric: Metric, reference_draws: ArrayLike, candidate_draws: ArrayLike
    ) -> list[MemorizationOutcome]:
        """Compute one requested point metric while preserving supplied source draws."""
        return [
            result[metric.name]
            for result in self.compute_memorization_group(
                [metric], reference_draws, candidate_draws
            )
        ]

    def compute_memorization_group(
        self,
        metrics: Sequence[Metric],
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        holdout_draws: ArrayLike | None = None,
    ) -> list[dict[str, MemorizationOutcome]]:
        """Share distance preparation and required searches on CPU or CUDA.

        Position matrices have equal batch lengths and may have unequal widths.
        Numerical executors preserve all supplied positions without sampling.
        """
        if not metrics or any(
            metric.name not in {"dcr", "near_match_rate", "replication_ratio"}
            or not self.has_kernel(metric, device=self.inputs.device.type, dtype=self.inputs.dtype)
            for metric in metrics
        ):
            raise UnsupportedExecutionError("Unsupported memorization metric group")
        if self._distances is None:
            from tabgen_eval.execution.torch.distance.executor import DistanceExecutor

            self._distances = DistanceExecutor(proxy(self))
        for metric in metrics:
            self._refresh_metric_memory(metric)
        return self._distances.compute_memorization_group(
            metrics, reference_draws, candidate_draws, holdout_draws=holdout_draws
        )

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
        """Delegate fold-preserving classifier fits and held-out scoring.

        Parameters
        ----------
        metric : ClassifierMetric
            Supported classifier and model/resource options.
        reference_draws, candidate_draws : array-like of int
            Equal-batch-length source-position matrices preserving bootstrap copies.
        reference_folds, candidate_folds : array-like of int
            Matching draw-shaped fold labels, stable for each original source row.
        seeds : iterable of int
            Exact uint64-range seeds, one per submitted draw.
        ideal : bool, default=False
            Use reference storage for both classes and validate shared source folds.
        point : bool, default=False
            Apply point-fit rather than bootstrap-fit work/thread limits.

        Returns
        -------
        list of ClassifierOutcome
            Ordered scores and fold diagnostics, including nonconvergence and
            numerical/resource failures. Training-only transforms and private solver
            RNG are owned by the executor; bootstrap/report decisions are not.
        """
        self._refresh_metric_memory(metric)
        return self._executor(metric).compute_classifier(
            metric,
            reference_draws,
            candidate_draws,
            reference_folds,
            candidate_folds,
            seeds,
            ideal=ideal,
            point=point,
            fit_threads=fit_threads,
        )
