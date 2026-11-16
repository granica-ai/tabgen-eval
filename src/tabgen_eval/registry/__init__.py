"""The declare-once metric registry."""

from .registry import (
    CATEGORY_DISPLAY_NAMES,
    REGISTRY,
    MetricSpec,
    get_by_name,
    resolve_metrics,
)
from .selection import (
    SELECTIONS,
    MetricSelection,
    build_metrics,
    list_groups,
    list_metrics,
    resolve_selection,
)

__all__ = [
    "REGISTRY",
    "MetricSpec",
    "CATEGORY_DISPLAY_NAMES",
    "get_by_name",
    "resolve_metrics",
    "list_metrics",
    "SELECTIONS",
    "MetricSelection",
    "list_groups",
    "resolve_selection",
    "build_metrics",
]
