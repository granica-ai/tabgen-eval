"""Internal computation adapters and their display metadata.

Public metric selection lives in ``selection``. Several public metrics share
one of these computation adapters so preparation and distance work are reused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .._options import ResolvedOptions
from ..metrics import (
    DCR,
    MLE,
    PRDC,
    Concentration,
    Conditional,
    LinearC2ST,
    Metric,
    NearMatchRate,
    NonLinearC2ST,
    ReplicationRatio,
    Shape,
    Trend,
)


@dataclass(frozen=True)
class MetricSpec:
    """Registry entry for one metric."""

    metric_cls: type[Metric]

    @property
    def name(self) -> str:
        return self.metric_cls.name

    @property
    def category(self) -> str:
        return self.metric_cls.category


# Order = display order.
REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec(Shape),
    MetricSpec(Trend),
    MetricSpec(PRDC),
    MetricSpec(Conditional),
    MetricSpec(Concentration),
    MetricSpec(LinearC2ST),
    MetricSpec(NonLinearC2ST),
    MetricSpec(DCR),
    MetricSpec(NearMatchRate),
    MetricSpec(ReplicationRatio),
    MetricSpec(MLE),
)

# Keyed by metric *category* (what summary() groups by), not by metric/stat name.
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "fidelity": "Fidelity (Distribution Similarity)",
    "conditional_fidelity": "Conditional (Subpopulation Fidelity)",
    "geometric_fidelity": "Geometric Fidelity",
    "detection": "Detection (Classifier Two-Sample Tests)",
    "memorization": "Memorization (Training-record Reproduction)",
    "utility": "Utility (Supervised Machine Learning)",
    "concentration": "Concentration (Local Neighborhood Mass)",
}

_BY_NAME: dict[str, MetricSpec] = {m.name: m for m in REGISTRY}


def get_by_name(name: str) -> MetricSpec | None:
    return _BY_NAME.get(name)


def resolve_metrics(selection: Sequence[str], options: ResolvedOptions) -> list[Metric]:
    """Instantiate explicitly resolved computation names in stable input order.

    Public metric/group validation and role compatibility live in ``selection``.
    This private adapter factory has no independent category or default policy.
    """
    chosen: list[Metric] = []
    for name in dict.fromkeys(selection):
        spec = get_by_name(name)
        if spec is None:
            raise ValueError(f"Unknown computation {name!r}. Available: {list(_BY_NAME)}")
        fixed = spec.metric_cls.fixed_execution_options
        metric_options = {name: options[name] for name in ("device", "dtype") if name not in fixed}
        chosen.append(spec.metric_cls(**metric_options))
    return chosen
