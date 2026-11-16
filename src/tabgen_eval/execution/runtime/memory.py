"""Automatic, evaluation-local allowances for managed numerical work.

Allowances are admission estimates, not allocator quotas. Inputs already resident
in a process are reflected in observed availability; only new source storage is
subtracted before placement. Private metric constructors may retain explicit
budgets for backend experiments and numerical regression tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, TypeVar, cast

if TYPE_CHECKING:
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.metrics.c2st import ClassifierMetric

_MetricT = TypeVar("_MetricT", bound="Metric")

MIB = 1024**2
_POLICY_NUMERATOR = 4
_POLICY_DENOMINATOR = 5


def _read_integer(path: str | PathLike[str]) -> int | None:
    try:
        value = Path(path).read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _cgroup_available_bytes() -> int | None:
    """Return the tightest visible Linux memory-controller ancestor headroom."""
    try:
        groups = Path("/proc/self/cgroup").read_text().splitlines()
        mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    limits: list[int] = []
    for group in groups:
        _, controllers, membership = group.split(":", 2)
        unified = not controllers
        if not unified and "memory" not in controllers.split(","):
            continue
        for line in mounts:
            fields, filesystem = line.split(" - ", 1)
            fields, filesystem = fields.split(), filesystem.split()
            if filesystem[0] != ("cgroup2" if unified else "cgroup"):
                continue
            if not unified and "memory" not in filesystem[-1].split(","):
                continue
            root, mount = Path(fields[3]), Path(fields[4])
            try:
                relative = Path(membership).relative_to(root)
            except ValueError:
                continue
            # A cgroup namespace may hide the membership prefix reported in
            # /proc; the visible controller root still constrains this process.
            current = mount if ".." in relative.parts else mount / relative
            while current == mount or mount in current.parents:
                maximum = _read_integer(
                    current / ("memory.max" if unified else "memory.limit_in_bytes")
                )
                used = _read_integer(
                    current / ("memory.current" if unified else "memory.usage_in_bytes")
                )
                if maximum is not None and used is not None:
                    limits.append(max(0, maximum - used))
                if current == mount:
                    break
                current = current.parent
    return min(limits) if limits else None


def host_available_bytes() -> int:
    """Observe available physical RAM, bounded by visible cgroup limits, without swap."""
    import psutil

    available = int(psutil.virtual_memory().available)
    constrained = _cgroup_available_bytes()
    return max(0, available if constrained is None else min(available, constrained))


def cuda_available_bytes(device: str) -> tuple[int, int, int]:
    """Observe driver-free bytes and report Torch cache without crediting it twice.

    Cached-but-unallocated Torch memory is not universally available to another
    allocator such as XGBoost. It is provenance, not additional shared capacity.
    No global allocator settings or caller-owned caches are changed.
    """
    from tabgen_eval.execution.torch.tensors import torch_module

    torch = torch_module(required=True)
    index = torch.device(device)
    free, _ = torch.cuda.mem_get_info(index)
    reserved, allocated = torch.cuda.memory_reserved(index), torch.cuda.memory_allocated(index)
    cached = max(0, reserved - allocated)
    # Split allocator blocks may be stranded beside live tensors. Excluding
    # these is conservative; remaining cache is still an estimate, not a
    # promise that a particular contiguous allocation will succeed.
    stats = torch.cuda.memory_stats(index)
    active = stats.get("active_bytes.all.current")
    fragmented = stats.get("inactive_split_bytes.all.current")
    # Active storage includes blocks awaiting another stream's completion. Do
    # not synchronize caller streams or count that pending storage as reusable.
    reusable = (
        0
        if active is None or fragmented is None
        else max(0, reserved - max(allocated, active) - fragmented)
    )
    return int(free), int(cached), int(reusable)


def automatic(metric: Metric, option: str) -> bool:
    """Whether an internal budget was omitted from the metric constructor."""
    return option not in getattr(metric, "_explicit_memory_options", ())


class MemoryPlanInfo(TypedDict):
    """Serializable capacity observations and managed per-task allowances."""

    policy: str
    available_host_bytes: int
    available_device_bytes: int | None
    cached_torch_bytes: int
    reusable_torch_cache_bytes: int
    new_source_bytes: int
    persistent_input_bytes: int
    owned_task_bytes: int
    aggregate_workspace_bytes: int
    task_workspace_bytes: int
    concurrent_tasks: int
    worker_overhead_bytes: int
    device: str


@dataclass(frozen=True)
class MemoryPlan:
    """One stage's observed capacity and managed per-task allowances, in bytes."""

    policy: str
    available_host_bytes: int
    available_device_bytes: int | None
    cached_torch_bytes: int
    reusable_torch_cache_bytes: int
    new_source_bytes: int
    persistent_input_bytes: int
    owned_task_bytes: int
    aggregate_workspace_bytes: int
    task_workspace_bytes: int
    concurrent_tasks: int
    worker_overhead_bytes: int
    device: str

    def to_dict(self) -> MemoryPlanInfo:
        return cast("MemoryPlanInfo", asdict(self))


class ResourcePlanner:
    """Resolve finite budgets from four fifths of current usable headroom.

    The fraction is this package's initial headroom heuristic, not a process
    quota or an industry standard. Actual allocations remain demand-driven.
    """

    def resolve(
        self,
        metric: Metric,
        device: str,
        *,
        new_source_bytes: int = 0,
        persistent_input_bytes: int = 0,
        owned_task_bytes: int = 0,
        concurrent_tasks: int = 1,
        minimum_task_bytes: int = 0,
        worker_overhead_bytes: int = 0,
    ) -> MemoryPlan:
        host = host_available_bytes()
        cuda = str(device).startswith("cuda")
        free: int | None = None
        cached, reusable = 0, 0
        available = host
        if cuda:
            free, cached, reusable = cuda_available_bytes(device)
            if getattr(metric, "execution_backend", None) == "xgboost":
                reusable = 0
            available = free + reusable
        if cuda and new_source_bytes > host * _POLICY_NUMERATOR // _POLICY_DENOMINATOR:
            raise MemoryError(
                f"Insufficient host memory to stage {new_source_bytes} source bytes "
                f"for {metric.name}: {host} physical bytes are available"
            )
        remaining = max(0, available - new_source_bytes)
        aggregate = remaining * _POLICY_NUMERATOR // _POLICY_DENOMINATOR
        lease = getattr(metric, "_memory_lease_bytes", None)
        if lease is not None:
            aggregate = min(aggregate, max(0, lease - owned_task_bytes))
        tasks = max(1, int(concurrent_tasks))
        if minimum_task_bytes:
            tasks = min(tasks, max(1, aggregate // (minimum_task_bytes + worker_overhead_bytes)))
        overhead = worker_overhead_bytes if tasks > 1 else 0
        per_task = max(0, aggregate // tasks - overhead)
        # MiB remains the private backend protocol. Do not introduce a minimum
        # allocation beyond the observed allowance when capacity is exhausted.
        if per_task < MIB and (
            automatic(metric, "gpu_workspace_mb")
            or (hasattr(metric, "memory_limit_mb") and automatic(metric, "memory_limit_mb"))
        ):
            raise MemoryError(
                f"Insufficient available memory for {metric.name}: "
                f"{remaining} bytes remain before the 80% workspace allowance"
            )
        if automatic(metric, "gpu_workspace_mb"):
            metric.gpu_workspace_mb = per_task // MIB
        if hasattr(metric, "memory_limit_mb") and automatic(metric, "memory_limit_mb"):
            metric.memory_limit_mb = per_task // MIB
        plan = MemoryPlan(
            "available_memory_80_percent",
            host,
            free,
            cached,
            reusable,
            int(new_source_bytes),
            int(persistent_input_bytes),
            int(owned_task_bytes),
            aggregate,
            per_task,
            tasks,
            int(overhead),
            str(device),
        )
        metric._memory_plan = plan.to_dict()
        metric._memory_concurrent_tasks = tasks
        return plan


def plan_metric_memory(
    metric: Metric,
    device: str = "cpu",
    *,
    new_source_bytes: int = 0,
    persistent_input_bytes: int = 0,
    owned_task_bytes: int = 0,
    concurrent_tasks: int = 1,
    minimum_task_bytes: int = 0,
    worker_overhead_bytes: int = 0,
) -> MemoryPlan:
    """Refresh a standalone metric's managed allowance at a stage boundary."""
    return ResourcePlanner().resolve(
        metric,
        device,
        new_source_bytes=new_source_bytes,
        persistent_input_bytes=persistent_input_bytes,
        owned_task_bytes=owned_task_bytes,
        concurrent_tasks=concurrent_tasks,
        minimum_task_bytes=minimum_task_bytes,
        worker_overhead_bytes=worker_overhead_bytes,
    )


def leased_metric(metric: _MetricT, allowance_bytes: int) -> _MetricT:
    """Copy an adapter for one process worker without permitting a larger allowance."""
    result = copy(metric)
    result._memory_lease_bytes = int(allowance_bytes)
    return result


def process_worker_bytes() -> int:
    """Conservative measured startup allowance for a separate Python worker.

    Parent RSS can include shared libraries and unrelated user data, so this
    deliberately overestimates some workers. Serial execution does not pay it.
    """
    import psutil

    return int(psutil.Process().memory_info().rss)


def xgboost_workspace_bytes(
    rows: int, numeric: int, cardinalities: Sequence[int], *, output_groups: int = 1
) -> int:
    """Estimate default hist fitting storage, including multiclass gradient work."""
    width = numeric + len(cardinalities)
    bins = numeric * 256 + sum(cardinalities) + len(cardinalities)
    return rows * (8 * width + 32 * output_groups) + bins * 128 * 16 * output_groups + 16 * MIB


def xgboost_model_bytes(output_groups: int = 1) -> int:
    """Conservative retained estimate for default 100-round depth-six trees."""
    return 16 * MIB + 100 * 127 * 64 * output_groups


def classifier_minimum_bytes(
    metric: ClassifierMetric,
    rows: int,
    numeric: int,
    categorical: int,
    cardinalities: Sequence[int],
) -> int:
    """Conservative compact-fit estimate used to limit simultaneous CPU fits.

    Fold-specific allocation guards remain authoritative after preprocessing.
    This estimate schedules work; it does not change solver thresholds or fits.
    """
    item = 4 if metric.dtype == "float32" else 8
    width = 2 * numeric + sum(min(card + 1, rows) for card in cardinalities) + categorical + 1
    gathered = rows * (numeric * item + categorical * 8 + 17)
    transformed = rows * (numeric * item * 4 + categorical * 16 + 64)
    rank = min(getattr(metric, "saga_rank", 64), width)
    batch = min(getattr(metric, "saga_batch_size", 4096), rows)
    calibration = min(4096, rows)
    pn = 2 * numeric + 1
    # Mirror the large terms in native saga::workspace_bytes: sixteen d*r
    # state buffers plus four d*r calibration buffers. Using every source row
    # rather than only the training fold keeps this scheduling estimate safe.
    solver = (
        item
        * (
            rows * pn
            + 2 * width
            + 2 * rows
            + 20 * width * rank
            + 24 * width
            + batch * (pn + 8)
            + calibration * (2 * pn + 2 * rank + 8)
        )
        + 8 * (rows + batch + calibration) * categorical
    )
    if getattr(metric, "saga_backoff", True):
        solver += item * (calibration * (pn + 9) + rows + 4 * width)
        solver += 8 * calibration * categorical
    return gathered + transformed + solver + 16 * MIB


class _CardinalitySource(Protocol):
    @property
    def cardinalities(self) -> Sequence[int]: ...


class _ClassifierMemoryContext(Protocol):
    """Only source widths, cardinalities, and metadata needed for admission."""

    n_numeric: int
    n_categorical: int

    @property
    def inputs(self) -> _CardinalitySource: ...

    @property
    def execution_info(self) -> dict[str, object]: ...


def plan_classifier_memory(
    metric: ClassifierMetric,
    context: _ClassifierMemoryContext,
    rows: int,
    *,
    concurrent_tasks: int = 1,
) -> MemoryPlan:
    """Share current host capacity across feasible concurrent classifier tasks."""
    minimum = classifier_minimum_bytes(
        metric, rows, context.n_numeric, context.n_categorical, context.inputs.cardinalities
    )
    if not automatic(metric, "memory_limit_mb"):
        minimum = max(minimum, int(metric.memory_limit_mb * MIB))
    info = context.execution_info
    return plan_metric_memory(
        metric,
        cast("str", info["device"]),
        persistent_input_bytes=cast("int", info.get("persistent_input_bytes", 0)),
        concurrent_tasks=concurrent_tasks,
        minimum_task_bytes=minimum,
    )
