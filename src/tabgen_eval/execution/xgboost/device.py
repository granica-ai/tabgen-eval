"""Select native XGBoost placement before preparing device-owned inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from tabgen_eval.execution.runtime.memory import (
    MIB,
    plan_metric_memory,
    xgboost_model_bytes,
    xgboost_workspace_bytes,
)
from tabgen_eval.models import XGBoostModel

if TYPE_CHECKING:
    from tabgen_eval.metrics import Metric


def cuda_device() -> tuple[str | None, str]:
    """Check the GPU engine and array bridge without changing the current device.

    Optional GPU dependencies may be absent or unusable in automatic mode.
    Missing XGBoost itself still raises, since CPU execution also requires it.
    The bridge probe allocates one scalar and never copies caller input data.
    """
    engine = XGBoostModel.load_backend()
    build_info = getattr(engine, "build_info", None)
    if build_info is None or not build_info().get("USE_CUDA", False):
        return None, "xgboost_cuda_unavailable"
    try:
        from tabgen_eval.execution.torch.tensors import torch_module

        torch = torch_module(required=False)
        if torch is None or not torch.cuda.is_available():
            return None, "cuda_unavailable"
        try:
            import cupy
        except ImportError:
            return None, "cupy_unavailable"
        if not callable(getattr(cupy, "from_dlpack", None)):
            return None, "cupy_bridge_unavailable"
        from tabgen_eval.execution.torch.context import cuda_supported, triton_installed

        if not triton_installed():
            return None, "triton_unavailable"
        if not cuda_supported(torch):
            return None, "cuda_version_below_minimum"
        index = torch.cuda.current_device()
        with cupy.cuda.Device(index):
            view = cupy.from_dlpack(torch.empty(1, device=f"cuda:{index}"))
            del view
        return f"cuda:{index}", "xgboost_cuda_available"
    except (ImportError, OSError, RuntimeError):
        return None, "cuda_environment_unavailable"


def select_device(
    metric: Metric,
    *,
    rows: int,
    numeric: int,
    cardinalities: Sequence[int],
    new_source_bytes: int = 0,
    output_groups: int = 1,
) -> str:
    """Prefer an available GPU in auto mode, with conservative memory admission.

    Explicit devices retain their strict dependency and memory errors. Selection
    precedes source transfers and is recorded separately from the requested
    device. Actual fits keep their normal statistical and failure protocols.
    """
    if metric.device != "auto":
        device, reason = XGBoostModel.execution_device(metric.device), "explicit_device"
        plan_metric_memory(metric, device, new_source_bytes=new_source_bytes)
    else:
        device, reason = cuda_device()
        if device is not None:
            try:
                plan = plan_metric_memory(metric, device, new_source_bytes=new_source_bytes)
                preparation = rows * (64 * numeric + 112 * len(cardinalities) + 32) + 65536
                required = (
                    preparation
                    + xgboost_workspace_bytes(
                        rows, numeric, cardinalities, output_groups=output_groups
                    )
                    + 2 * xgboost_model_bytes(output_groups)
                )
                budget = min(
                    plan.task_workspace_bytes,
                    metric.gpu_workspace_mb * MIB,
                    getattr(metric, "memory_limit_mb", metric.gpu_workspace_mb) * MIB,
                )
                if required > budget:
                    device, reason = None, "xgboost_workspace_too_small"
            except MemoryError:
                device, reason = None, "insufficient_free_cuda_memory"
            except (OSError, RuntimeError):
                device, reason = None, "cuda_environment_unavailable"
        if device is None:
            device = "cpu"
            plan_metric_memory(metric, device)
    metric._xgboost_device = device
    metric._xgboost_routing_reason = reason
    return device


def resolved_device(metric: Metric) -> str:
    """Read a recorded placement without probing dependencies during reporting."""
    if hasattr(metric, "_xgboost_device"):
        return metric._xgboost_device
    # A skipped metric has no placement. Reporting must neither choose one nor
    # initialize CUDA just to canonicalize an unindexed request.
    return "unresolved" if metric.device in {"auto", "cuda"} else metric.device
