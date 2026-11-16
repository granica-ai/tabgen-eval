"""Component selection for shared numerical jobs.

Public selectors may share one metric implementation. These helpers retain a
stable component order for the Python and native execution adapters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabgen_eval.metrics.base import Metric

CONDITIONAL_COMPONENTS = ("cond_shape", "cond_trend", "recovery")
PRDC_COMPONENTS = ("precision", "recall", "density", "coverage")


def selected_components(
    requested: Iterable[str] | None, available: Sequence[str]
) -> tuple[str, ...]:
    """Validate a component selection and return it in implementation order."""
    if requested is None:
        return tuple(available)
    if isinstance(requested, str):
        raise ValueError("requested_components must be a sequence of component names")
    requested = tuple(requested)
    unknown = set(requested).difference(available)
    if unknown:
        raise ValueError(
            f"Unknown components {sorted(unknown)!r}; valid components are {list(available)!r}"
        )
    return tuple(name for name in available if name in requested)


def component_mask(metric: Metric) -> int:
    """Encode selected components for the native sample/detailed operators."""
    if metric.name == "conditional_fidelity":
        available = CONDITIONAL_COMPONENTS
    elif metric.name == "prdc":
        available = PRDC_COMPONENTS
    else:
        return 15
    selected = selected_components(metric.requested_components, available)
    return sum(1 << index for index, name in enumerate(available) if name in selected)
