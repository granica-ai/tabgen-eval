"""Figures from captured column, pair and conditional-slice tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from ._data import detail_table, select_rows
from ._style import COLORS, category_label, finish, status, text

if TYPE_CHECKING:
    from tabgen_eval.report import Report


def columns(
    report: Report, *, metric: str = "shape", stat: str | None = None, top_n: int | None = None
) -> go.Figure:
    """Plot stored column scores, with explicit labels for undefined outcomes."""
    frame = detail_table(report, metric, "column")
    stat = stat or {
        "conditional_fidelity": "cond_shape_all",
        "cond_shape": "cond_shape_all",
        "cond_trend": "cond_trend_all",
        "recovery": "recovery_all",
    }.get(metric, "score")
    frame = select_rows(frame, stat, top_n)
    label_column = "column" if "column" in frame else "target"
    labels = [text(value) for value in frame[label_column]]
    values = frame[stat].to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=[value if np.isfinite(value) else None for value in values],
            y=labels,
            orientation="h",
            marker_color=COLORS["generated"],
            hovertemplate="%{y}: %{x:.5g}<extra></extra>",
        )
    )
    for label, value in zip(labels, values, strict=True):
        if not np.isfinite(value):
            fig.add_annotation(
                x=0,
                xref="paper",
                y=label,
                yref="y",
                text=f"{label}: {status(value)}",
                showarrow=False,
                xanchor="left",
            )
    fig.update_xaxes(title_text=text(stat))
    fig.update_yaxes(
        autorange="reversed", type="category", categoryorder="array", categoryarray=labels
    )
    return finish(
        fig,
        text(f"{metric} · {stat} by column"),
        height=max(350, 90 + len(frame) * 28),
        metadata={
            "kind": "columns",
            "metric": metric,
            "stat": stat,
            "uncertainty": "none; descriptive point scores",
        },
    )


def pairs(
    report: Report, *, metric: str = "trend", columns: Sequence[str] | None = None
) -> go.Figure:
    """Plot stored pair scores; absent/undefined pairs remain unfilled."""
    frame = detail_table(report, metric, "pair")
    labels = list(dict.fromkeys([*frame["column_1"], *frame["column_2"]]))
    if columns is not None:
        selected = list(dict.fromkeys(columns))
        unknown = set(selected) - set(labels)
        if unknown:
            raise KeyError(f"Columns absent from stored pairs: {unknown}")
        if len(selected) < 2:
            raise ValueError("Select at least two columns for a pair heatmap")
        labels = selected
        frame = frame.loc[frame["column_1"].isin(labels) & frame["column_2"].isin(labels)]
    lookup = {name: index for index, name in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan)
    for first, second, score in frame[["column_1", "column_2", "score"]].itertuples(
        index=False, name=None
    ):
        i, j = lookup[first], lookup[second]
        matrix[i, j] = matrix[j, i] = score
    display = [text(label) for label in labels]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.tolist(),
            x=display,
            y=display,
            zmin=0,
            zmax=1,
            colorscale="Viridis",
            colorbar={"title": "Score"},
            hoverongaps=False,
            hovertemplate="%{y} × %{x}: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(plot_bgcolor="#e9edf0")
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    return finish(
        fig,
        text(f"{metric} · pair scores (blank: undefined or absent)"),
        height=max(430, min(900, 220 + len(labels) * 26)),
        metadata={"kind": "pairs", "metric": metric, "missing": "unfilled"},
    )


def slices(
    report: Report,
    *,
    column: str,
    metric: str | None = None,
    stat: str = "shape",
    top_n: int | None = None,
) -> go.Figure:
    """Plot one conditioning column's stored slice scores and eligibility.

    Hover text retains counts and rarity. Ineligible or undefined rows are
    explicitly labelled; no slice-level uncertainty is estimated.
    """
    if stat not in {"shape", "trend", "recovery"}:
        raise ValueError("slice stat must be 'shape', 'trend', or 'recovery'")
    if metric is None:
        public_name = {"shape": "cond_shape", "trend": "cond_trend", "recovery": "recovery"}[stat]
        names = {result.name for result in report.results}
        metric = public_name if public_name in names else "conditional_fidelity"
    frame = detail_table(report, metric, "slice")
    frame = frame.loc[frame["column"].map(lambda value: value == column)].copy()
    if frame.empty:
        raise KeyError(f"No stored slices for conditioning column {column!r}")
    frame = select_rows(frame, stat, top_n)
    labels = [
        text(str(value)) if hasattr(value, "closed") else category_label(value)
        for value in frame["label"]
    ]
    values = frame[stat].to_numpy(dtype=float)
    eligible = frame["eligible"].to_numpy(dtype=bool)
    visible = np.isfinite(values) & eligible
    custom = frame[["reference_count", "candidate_count", "eligible", "rare"]].to_numpy().tolist()
    fig = go.Figure(
        go.Bar(
            x=[value if keep else None for value, keep in zip(values, visible, strict=True)],
            y=labels,
            orientation="h",
            customdata=custom,
            marker_color=COLORS["generated"],
            hovertemplate=(
                "%{y}<br>Score: %{x:.5g}<br>Real count: %{customdata[0]}"
                "<br>Generated count: %{customdata[1]}<br>Eligible: %{customdata[2]}"
                "<br>Rare: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    for label, value, keep, counts in zip(labels, values, visible, custom, strict=True):
        if not keep:
            reason = "ineligible" if not counts[2] else status(value)
            fig.add_annotation(
                text=f"{reason}; real={counts[0]}, generated={counts[1]}",
                x=0,
                xref="paper",
                y=label,
                yref="y",
                xanchor="left",
                showarrow=False,
            )
    fig.update_xaxes(title_text=text(stat))
    fig.update_yaxes(
        type="category", autorange="reversed", categoryorder="array", categoryarray=labels
    )
    return finish(
        fig,
        text(f"Conditional {stat} · {column}"),
        height=max(350, 100 + 31 * len(frame)),
        metadata={
            "kind": "slices",
            "metric": metric,
            "column": str(column),
            "stat": stat,
            "uncertainty": "none; descriptive point scores",
        },
    )
