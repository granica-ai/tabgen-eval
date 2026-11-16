"""Result dataclasses: ``StatResult`` (one scalar) and ``MetricResult`` (one metric).

Scalar results preserve point values, raw replicate distributions and factual diagnostics. ``Report`` assembles their presentation views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from tabgen_eval.execution.runtime.details import ConcentrationDetails, FidelityDetails

from ..metrics.base import Direction
from ..report.diagnostics import DiagnosticFlag
from ..statistics.distributions import ScoreDistribution

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.domains import RowUsage

NAN = float("nan")


@dataclass
class StatResult:
    """A point score with optional raw score distributions and factual diagnostics.

    For paired holdout statistics, ``value`` and ``ideal`` are generated-real
    and real-real scores. A side with repeated comparisons reports their finite
    mean; an unrepeated side reports a single score. ``value_estimate`` and
    ``ideal_estimate`` distinguish these cases. MLE
    retains its raw real-trained AUROC or MAE as ideal; its ratio/prediction
    statistics and memorization statistics have no artificial ideal.
    Repeated scores and their summaries live in their respective
    distribution objects. Interval fields contain unshifted central percentile
    bands, with NaN endpoints when a distribution was not requested or has no
    finite scores. No automatic comparison labels are attached.

    Diagnostics record applicability, numerical and fitting conditions without
    inferring universal reliability. Ordered samples and discarded counts are
    retained in the distribution objects; export controls determine whether raw
    samples appear in serialized views.

    Test-only repeated evaluation stores its scores in ``reference_distribution``
    and reports its mean in ``value``. Its ``score_role`` distinguishes
    test-test comparisons from real-trained predictive scores. A single actual
    test-only point uses ``value`` without an artificial generated score or ideal.
    """

    name: str
    display: str
    value: float
    ci_low: float
    ci_high: float
    bootstrap_median: float
    ideal: float
    ideal_ci_low: float
    ideal_ci_high: float
    ideal_bootstrap_median: float
    direction: Direction
    target: float | None
    n_holdout: int
    n_generated: int
    n_eff: int
    """Rows selected per comparison role, before metric-specific observed-value handling."""
    flags: list[DiagnosticFlag] = field(default_factory=list[DiagnosticFlag])
    fmt: str = ".4f"
    generated_distribution: ScoreDistribution | None = None
    reference_distribution: ScoreDistribution | None = None
    score_role: str = "generated_test"
    """Source of the primary score, including ``test_test`` and ``real_trained``."""
    point_status: str | None = None
    ideal_status: str | None = None
    """Explicit ``absent`` markers distinguish uncomputed scores from undefined ones."""
    value_estimate: str = "point"
    ideal_estimate: str = "point"
    """Whether each reported score is a single comparison or a repetition mean."""

    @property
    def flag_codes(self) -> list[str]:
        return [f.code for f in self.flags]


@dataclass
class MetricResult:
    """Named scalar results with shared execution diagnostics and fidelity details."""

    name: str
    category: str
    stats: dict[str, StatResult] = field(default_factory=dict[str, StatResult])
    diagnostics: dict[str, object] = field(default_factory=dict[str, object])
    """Optional fitted-metric diagnostics, serialized in report provenance."""
    details: FidelityDetails | ConcentrationDetails | None = None
    primary: str | None = None
    """Statistic returned when a public metric name is used as a report lookup."""
    computation: str | None = None
    """Internal computation that produced this public metric's selected statistics."""

    def __getitem__(self, stat: str) -> StatResult:
        return self.stats[stat]

    @property
    def row_usage(self) -> RowUsage | None:
        """Eligibility metadata attached by the shared sampling planner.

        Backend diagnostics remain extensible; the planner owns and validates
        this particular record before storing it alongside those diagnostics.
        """
        return cast("RowUsage | None", self.diagnostics.get("row_usage"))
