"""Assemble training-referenced metric values and factual diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from tabgen_eval.execution.python.stats import safe_ratio
from tabgen_eval.execution.runtime.outcomes import DCR_PERCENTILES, MemorizationOutcome

if TYPE_CHECKING:
    from tabgen_eval.metrics.memorization import _MemorizationMetric

NEAR_MATCH_THRESHOLD = 1e-3


def memorization_metric_outcome(
    metric: _MemorizationMetric,
    *,
    counts: Sequence[int],
    median: float = np.nan,
    q05: float = np.nan,
    holdout_median: float = np.nan,
    generated_quantiles: Mapping[int, float] | None = None,
    holdout_quantiles: Mapping[int, float] | None = None,
    holdout_count: int | None = None,
    rate: float = np.nan,
    diagnostics: Mapping[str, object] | None = None,
) -> MemorizationOutcome:
    """Assemble one independently selectable training-referenced point metric.

    Numerical executors supply only requested reductions. Counts refer to usable
    rows after complete-row filtering; source sampling remains with the caller.
    """
    usable: dict[str, int | None] = dict(
        zip(("training", "generated"), map(int, counts), strict=True)
    )
    details: dict[str, object] = {
        "usable_rows": usable,
        **(diagnostics or {}),
    }
    flags: dict[str, list[str]] = {}
    if metric.name == "dcr":
        usable["holdout"] = holdout_count
        zero = bool(holdout_median == 0.0)
        if holdout_count is None:
            flags["ratio"] = ["MISSING_HOLDOUT"]
        elif not holdout_count:
            flags["ratio"] = ["INSUFFICIENT_HOLDOUT_ROWS"]
        elif not counts[0]:
            flags["ratio"] = ["INSUFFICIENT_TRAINING_ROWS"]
        elif zero:
            flags["ratio"] = ["ZERO_BASELINE"]
        values = {
            "median": float(median),
            "q05": float(q05),
            "ratio": safe_ratio(median, holdout_median),
        }
        values.update(
            {
                f"q{p:02d}": float(
                    (generated_quantiles or {}).get(
                        p, q05 if p == 5 else median if p == 50 else np.nan
                    )
                )
                for p in DCR_PERCENTILES
            }
        )
        values.update(
            {
                f"holdout_q{p:02d}": float(
                    (holdout_quantiles or {}).get(p, holdout_median if p == 50 else np.nan)
                )
                for p in DCR_PERCENTILES
            }
        )
        for p in DCR_PERCENTILES:
            key = f"holdout_q{p:02d}"
            if not np.isfinite(values[key]) and "ratio" in flags and not zero:
                flags[key] = list(flags["ratio"])
        details.update(
            holdout_median=float(holdout_median) if np.isfinite(holdout_median) else None,
            baseline="holdout_to_training",
            zero_baseline=zero,
        )
    elif metric.name in {"near_match_rate", "replication_ratio"}:
        values = {"rate": float(rate)}
        details.update(
            near_match_threshold=NEAR_MATCH_THRESHOLD,
            denominator=int(counts[1] if metric.name == "near_match_rate" else counts[0]),
            query_role="generated" if metric.name == "near_match_rate" else "training",
        )
    else:
        raise ValueError(f"Unsupported memorization metric {metric.name!r}")
    return MemorizationOutcome(values, details, flags)
