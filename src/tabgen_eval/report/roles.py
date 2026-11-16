"""Shared presentation names and distribution selection for stored score roles."""

from __future__ import annotations

from ..statistics.distributions import ScoreDistribution
from .result import StatResult

_LABELS = {
    "generated_test": "Generated–test",
    "generated_train": "Generated → train",
    "test_train": "Test → train",
    "train_generated": "Train → generated",
    "distance_ratio": "Generated / test distance",
    "test_test": "Test–test",
    "real_trained": "Real-trained",
}


def score_role_label(stat: StatResult) -> str:
    """Describe which observations or comparison produced the primary score."""
    return _LABELS.get(stat.score_role, stat.score_role)


def reference_only(stat: StatResult) -> bool:
    """Whether a statistic describes real-only evaluation without a generator."""
    return stat.score_role in {"test_test", "real_trained"}


def primary_distribution(stat: StatResult) -> ScoreDistribution | None:
    """Return the primary role's actual distribution, without inventing scores."""
    return stat.reference_distribution if reference_only(stat) else stat.generated_distribution
