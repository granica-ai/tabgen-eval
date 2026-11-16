"""Present source eligibility separately from sampled rows and finite scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .result import MetricResult

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.domains import RowUsage

_COLUMNS = [
    "group",
    "metric",
    "computation",
    "policy",
    "columns",
    "pool",
    "input",
    "retained",
    "excluded",
    "excluded_percent",
    "n_res",
]


def row_usage_frame(results: list[MetricResult]) -> pd.DataFrame:
    """Return one row per public metric and source pool, including zero exclusions."""
    rows: list[dict[str, object]] = []
    for result in results:
        usage = result.row_usage
        if usage is None:
            continue
        for pool, counts in usage["pools"].items():
            rows.append(
                {
                    "group": result.category,
                    "metric": result.name,
                    "computation": result.computation or result.name,
                    "policy": usage["policy"],
                    "columns": tuple(usage["columns"]),
                    "pool": pool,
                    **counts,
                    "n_res": usage["n_res"],
                }
            )
    return pd.DataFrame(rows, columns=_COLUMNS)


def _shared_usage(results: list[MetricResult]) -> list[tuple[list[str], RowUsage]]:
    """Group public outputs that share both their computation and eligibility."""
    groups: list[tuple[list[str], str, RowUsage]] = []
    for result in results:
        usage = result.row_usage
        if usage is None:
            continue
        computation = result.computation or result.name
        for names, previous_computation, previous in groups:
            if previous_computation == computation and previous == usage:
                names.append(result.name)
                break
        else:
            groups.append(([result.name], computation, usage))
    return [(names, usage) for names, _, usage in groups]


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"


def row_usage_summary(results: list[MetricResult]) -> list[str]:
    """Render source-level exclusions once per shared numerical computation."""
    groups = _shared_usage(results)
    if not groups:
        return []
    lines = ["row eligibility (before sampling):"]
    for names, usage in groups:
        policy = {
            "complete_case": "complete cases",
            "observed_target": "observed target",
        }[usage["policy"]]
        lines.append(f"  {', '.join(names)} ({policy}):")
        for pool, counts in usage["pools"].items():
            lines.append(
                f"    {pool}: {counts['retained']}/{counts['input']} retained; "
                f"excluded {counts['excluded']} ({_percent(counts['excluded_percent'])})"
            )
    return lines


def row_usage_markdown(results: list[MetricResult]) -> str:
    """Render source eligibility with unambiguous count and percentage denominators."""
    groups = _shared_usage(results)
    if not groups:
        return ""
    lines = [
        "**Row eligibility before sampling**",
        "",
        "| Metrics | Policy | Pool | Input rows | Retained rows | Excluded rows | Excluded % |",
        "|---|---|---|---|---|---|---|",
    ]
    for names, usage in groups:
        policy = {
            "complete_case": "Complete cases",
            "observed_target": "Observed target",
        }[usage["policy"]]
        for pool, counts in usage["pools"].items():
            lines.append(
                f"| {', '.join(names)} | {policy} | {pool} | {counts['input']} | "
                f"{counts['retained']} | {counts['excluded']} | "
                f"{_percent(counts['excluded_percent'])} |"
            )
    return "\n".join(lines)
