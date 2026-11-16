"""Adapters for stored results and captured descriptive tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._style import positive_integer, text

if TYPE_CHECKING:
    import pandas as pd

    from tabgen_eval.report import Report
    from tabgen_eval.report.result import MetricResult
    from tabgen_eval.statistics.distributions import ScoreDistribution


def metric_results(
    report: Report, metrics: str | Sequence[str] | None = None
) -> list[MetricResult]:
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        return list(report.results)
    requested = list(dict.fromkeys(metrics))
    by_name = {result.name: result for result in report.results}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise KeyError(f"Metrics not present in report: {missing}")
    return [by_name[name] for name in requested]


def detail_table(report: Report, metric: str, level: str) -> pd.DataFrame:
    frame = report.details(metric, level=level)
    if frame.empty:
        raise ValueError(f"No {level} details available for {metric!r}")
    return frame


def select_rows(frame: pd.DataFrame, stat: str, top_n: int | None) -> pd.DataFrame:
    """Sort finite values without turning undefined entries into zero scores."""
    if stat not in frame:
        raise ValueError(
            f"Statistic {stat!r} unavailable; available columns: {list(frame.columns)}"
        )
    frame = frame.sort_values(stat, kind="stable", na_position="last")
    if top_n is not None:
        frame = frame.head(positive_integer(top_n, "top_n"))
    return frame


def stored_band(distribution: ScoreDistribution | None) -> tuple[float, float] | None:
    """Only actual retained score distributions can contribute a score band."""
    if distribution is None or not distribution.finite_count:
        return None
    low, high = distribution.band_low, distribution.band_high
    return (float(low), float(high)) if np.isfinite(low) and np.isfinite(high) else None


def row_usage_hover(result: MetricResult) -> str:
    """Source exclusions for hover text, kept out of subplot titles and labels."""
    usage = result.row_usage
    if usage is None:
        return ""
    pieces = ["Row eligibility before sampling:"]
    for pool, counts in usage["pools"].items():
        percent = counts["excluded_percent"]
        rate = "undefined" if percent is None else f"{percent:.2f}%"
        pieces.append(
            f"{text(pool)}: {counts['retained']}/{counts['input']} retained; "
            f"excluded {counts['excluded']} ({rate})"
        )
    return "<br>".join(pieces)
