"""Raw numerical outcomes shared by metric adapters and estimators.

These containers hold statistic values, baselines, flags, and fit diagnostics.
They perform no sampling, computation, or report rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .details import ConcentrationDetails, FidelityDetails

DCR_PERCENTILES = (5, 10, 25, 50, 75, 90, 95, 99)


@dataclass
class ClassifierOutcome:
    """A raw score plus fold diagnostics; optimizer failure is not silently discarded."""

    value: float
    folds: list[dict[str, object]]

    @property
    def converged(self) -> bool:
        """Whether every recorded fold converged; false when no fold results exist."""
        return bool(self.folds) and all(f.get("converged", False) for f in self.folds)


@dataclass
class MemorizationOutcome:
    """Fixed training/generated statistics with diagnostics and per-statistic flags."""

    values: dict[str, float]
    diagnostics: dict[str, object] = field(default_factory=dict)
    stat_flags: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedMetricResult:
    """Paired point scores with execution diagnostics and optional descriptive tables.

    Both comparisons share one real reference draw. The real candidate is
    disjoint from that reference; details describe the generated comparison.
    """

    name: str
    values: dict[str, float]
    real_reference: dict[str, float]
    execution: dict[str, object]
    diagnostics: dict[str, object] = field(default_factory=dict)
    details: FidelityDetails | ConcentrationDetails | None = None


@dataclass
class UtilityOutcome:
    """Raw MLE scores, a paired real-trained score, and fit diagnostics."""

    values: dict[str, float]
    reference_scores: dict[str, float]
    diagnostics: dict[str, object] = field(default_factory=dict)
    stat_flags: dict[str, list[str]] = field(default_factory=dict)
