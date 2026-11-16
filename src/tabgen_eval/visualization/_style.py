"""Shared, renderer-independent figure styling and value labels."""

from __future__ import annotations

import html
import math
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from tabgen_eval.preprocessing.categories import category_key

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

COLORS = {"generated": "#2b70b5", "reference": "#d27622", "muted": "#77828c"}


def text(value: object) -> str:
    """Escape data labels before passing them to Plotly's HTML-aware text fields."""
    return html.escape(str(value), quote=False)


def category_label(value: object) -> str:
    """Distinguish scalar families, whitespace, and actual missing values."""
    key = category_key(value)
    if key is None or key[0] == "missing-category":
        return "〈missing〉"
    return text(f"{key[0]}: {value!r}")


def status(value: float) -> str:
    if np.isnan(value):
        return "undefined"
    return "+∞" if value > 0 else "−∞"


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def wrapped_text(value: object, *, width: int = 30) -> str:
    """Wrap plain labels before escaping, preserving their full content."""
    return "<br>".join(text(line) for line in textwrap.wrap(str(value), width=width))


def panel_title(value: object, subtitle: object = None) -> str:
    """Keep a panel's name and interpretation on separate, bounded lines."""
    heading = f"<b>{wrapped_text(value, width=26)}</b>"
    return heading if subtitle is None else f"{heading}<br>{wrapped_text(subtitle)}"


def panel_grid(
    titles: Sequence[str], *, columns: int = 2, panel_height: int = 120
) -> tuple[Figure, dict[str, int]]:
    """Allocate pixel space for panel headings, ticks and axis titles.

    Plotly's default fractional gutters shrink as more rows are added. Here
    every row retains its own heading and axis-label space, even in a large
    report. Two columns leave enough room for wrapped category labels.
    """
    from plotly.subplots import make_subplots

    columns = min(columns, len(titles))
    rows = math.ceil(len(titles) / columns)
    heading_height = max(title.count("<br>") + 1 for title in titles) * 19
    row_gap = heading_height + 70
    plot_height = rows * panel_height + (rows - 1) * row_gap
    fig = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=titles,
        horizontal_spacing=0.28 if columns > 1 else 0,
        vertical_spacing=row_gap / plot_height if rows > 1 else 0,
    )
    annotations = iter(fig.layout.annotations)
    for index, title in enumerate(titles):
        if not title:
            continue
        annotation = next(annotations)
        # Align to the plot rather than centering text over neighboring labels.
        axis = fig.layout["xaxis" if index == 0 else f"xaxis{index + 1}"]
        annotation.update(
            x=axis.domain[0], xanchor="left", align="left", yshift=12, font={"size": 13}
        )
    fig.update_xaxes(nticks=4, tickangle=0)
    return fig, {"plot_height": plot_height, "top": heading_height + 65}


def finish(
    fig: Figure,
    title: str,
    *,
    height: int = 430,
    plot_height: int | None = None,
    top: int = 85,
    metadata: dict[str, object] | None = None,
) -> Figure:
    """Apply a light, readable theme without setting a renderer or opening a browser."""
    has_legend = any(trace.name and trace.showlegend is not False for trace in fig.data)
    legend_size = sum(
        len(trace.name) + 7 for trace in fig.data if trace.name and trace.showlegend is not False
    )
    bottom = 100 + 25 * max(1, math.ceil(legend_size / 70)) if has_legend else 65
    if plot_height is None:
        plot_height = max(1, height - top - bottom)
    else:
        height = plot_height + top + bottom
    fig.update_layout(
        title={"text": title, "x": 0.03},
        template="plotly_white",
        height=height,
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26323d"},
        margin={"l": 70, "r": 35, "t": top, "b": bottom},
        legend={"orientation": "h", "yanchor": "top", "y": -60 / plot_height, "x": 0},
        meta=metadata or {},
    )
    fig.update_xaxes(
        automargin=True,
        title_standoff=14,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#dce3e8",
        ticks="outside",
        tickcolor="#dce3e8",
        ticklen=5,
    )
    fig.update_yaxes(automargin=True, showgrid=False, zeroline=False)
    return fig
