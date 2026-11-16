"""Describe one metric's resolved numerical implementation without running it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabgen_eval.metrics.base import Metric

    from .context import ContextMetadata


def execution_info(metric: Metric, context: ContextMetadata | None = None) -> dict[str, object]:
    """Own execution metadata and include the active automatic memory plan.

    Prepared contexts describe their selected implementation directly. Metrics
    executing through Python or XGBoost without a context use their declared
    implementation and resolved device. Metadata collection does not fit models.
    """
    if context is not None:
        return context.execution_info_for(metric)
    backend = getattr(metric, "execution_backend", "python")
    if backend == "xgboost":
        from tabgen_eval.execution.xgboost.device import resolved_device

        device = resolved_device(metric)
    else:
        device = "cpu"
    info: dict[str, object] = {"implementation": backend, "device": device}
    if backend == "xgboost" and hasattr(metric, "_xgboost_routing_reason"):
        info["routing_reason"] = metric._xgboost_routing_reason
    if hasattr(metric, "_memory_plan"):
        info["memory_plan"] = dict(metric._memory_plan)
    return info
