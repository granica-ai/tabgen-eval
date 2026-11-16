"""Bounded exploratory plots using the package's schema and category identities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from tabgen_eval.preprocessing.categories import factorize_categories
from tabgen_eval.preprocessing.columns import numeric_values
from tabgen_eval.schema import Schema, validate

from ._style import COLORS, category_label, finish, positive_integer, text

_Scalar = TypeVar("_Scalar", bound=np.generic)


def _prepare(
    real: pd.DataFrame, generated: pd.DataFrame, columns: Sequence[str], schema: Schema
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], Schema]:
    """Validate the plotted columns without changing their values or identities."""
    if not isinstance(real, pd.DataFrame) or not isinstance(generated, pd.DataFrame):
        raise TypeError("real_data and generated_data must be pandas DataFrames")
    for frame in (real, generated):
        missing = [name for name in columns if name not in frame]
        if missing:
            raise KeyError(f"Missing plotted columns: {missing}")
        if not frame.columns.is_unique:
            raise ValueError("Plot input columns must be unique")
    if not isinstance(schema, Schema):
        raise TypeError("schema must be a Schema; use Schema.infer(...) explicitly if desired")
    roles = set(schema.numeric + schema.categorical)
    if any(column not in roles for column in columns):
        raise ValueError("Every plotted column must have a declared schema role")
    selected = Schema(
        numeric=[c for c in columns if c in schema.numeric],
        categorical=[c for c in columns if c in schema.categorical],
        target=schema.target if schema.target in columns else None,
    )
    first, second = real.loc[:, columns], generated.loc[:, columns]
    validate(
        first,
        second,
        selected,
        allow_empty=True,
        reference_name="real_data",
        candidate_name="generated_data",
    )
    return (first, second), selected


def _sample(values: NDArray[_Scalar], limit: int, seed: int) -> NDArray[_Scalar]:
    if len(values) <= limit:
        return values
    positions = np.sort(np.random.default_rng(seed).choice(len(values), limit, replace=False))
    return values[positions]


def _categorical(
    first: pd.Series, second: pd.Series, limit: int
) -> tuple[list[NDArray[np.int64]], list[str]]:
    codes, vocabulary = factorize_categories(first, second)
    counts = np.bincount(
        np.concatenate([array[array >= 0] for array in codes]), minlength=len(vocabulary)
    )
    selected = np.argsort(-counts, kind="stable")[:limit]
    mapping = np.full(len(vocabulary), len(selected), dtype=int)
    mapping[selected] = np.arange(len(selected))
    labels = [category_label(vocabulary.values[index]) for index in selected]
    if len(selected) < len(vocabulary):
        labels.append("〈other categories〉")
    result = []
    for array in codes:
        encoded = np.full(len(array), -1, dtype=int)
        present = array >= 0
        encoded[present] = mapping[array[present]]
        result.append(encoded)
    return result, labels


def column_figure(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    column: str,
    *,
    schema: Schema,
    max_points: int,
    max_categories: int,
    seed: int,
) -> go.Figure:
    max_points = positive_integer(max_points, "max_points")
    max_categories = positive_integer(max_categories, "max_categories")
    frames, schema = _prepare(real, generated, [column], schema)
    fig = go.Figure()
    meta = {
        "kind": "column",
        "column": str(column),
        "sources": {},
    }
    if column in schema.numeric:
        for index, (frame, role, label) in enumerate(
            zip(frames, ("reference", "generated"), ("Real", "Generated"), strict=True)
        ):
            raw = numeric_values(frame[column])
            finite = raw[np.isfinite(raw)]
            values = np.sort(_sample(finite, max_points, seed + index))
            fig.add_trace(
                go.Scatter(
                    x=values.tolist(),
                    y=(np.arange(1, len(values) + 1) / len(values)).tolist() if len(values) else [],
                    mode="lines",
                    name=label,
                    line={"shape": "hv", "color": COLORS[role]},
                    hovertemplate="%{x:.5g}<br>Cumulative fraction: %{y:.4f}<extra></extra>",
                )
            )
            meta["sources"][role] = {
                "rows": len(frame),
                "missing_or_nonfinite": len(raw) - len(finite),
                "plotted": len(values),
                "available": len(finite),
            }
        fig.update_xaxes(title_text=text(column))
        fig.update_yaxes(title_text="Empirical cumulative fraction")
    else:
        codes, labels = _categorical(frames[0][column], frames[1][column], max_categories)
        for frame, array, role, label in zip(
            frames, codes, ("reference", "generated"), ("Real", "Generated"), strict=True
        ):
            count = np.bincount(array[array >= 0], minlength=len(labels))
            total = int(count.sum())
            values = count / total if total else np.full(len(count), np.nan)
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values.tolist(),
                    customdata=count.tolist(),
                    name=label,
                    marker_color=COLORS[role],
                    hovertemplate="%{x}<br>Fraction: %{y:.4f}<br>Count: %{customdata}<extra></extra>",
                )
            )
            meta["sources"][role] = {
                "rows": len(frame),
                "missing_or_nonfinite": len(array) - total,
                "plotted": total,
                "available": total,
            }
        fig.update_layout(barmode="group")
        fig.update_xaxes(
            title_text=text(column), type="category", categoryorder="array", categoryarray=labels
        )
        fig.update_yaxes(title_text="Fraction of observed values")
    return finish(fig, text(f"{column} · real and generated distributions"), metadata=meta)


def pair_figure(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    column_1: str,
    column_2: str,
    *,
    schema: Schema,
    max_points: int,
    max_categories: int,
    seed: int,
) -> go.Figure:
    if column_1 == column_2:
        raise ValueError("Choose two distinct columns for a pair plot")
    max_points = positive_integer(max_points, "max_points")
    max_categories = positive_integer(max_categories, "max_categories")
    frames, schema = _prepare(real, generated, [column_1, column_2], schema)
    numeric = [column for column in (column_1, column_2) if column in schema.numeric]
    meta = {
        "kind": "column_pair",
        "columns": [str(column_1), str(column_2)],
        "sources": {},
    }
    if len(numeric) == 2:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Real", "Generated"],
            shared_xaxes=True,
            shared_yaxes=True,
        )
        for index, (frame, role, label) in enumerate(
            zip(frames, ("reference", "generated"), ("Real", "Generated"), strict=True)
        ):
            values = np.column_stack([numeric_values(frame[c]) for c in (column_1, column_2)])
            complete = values[np.isfinite(values).all(axis=1)]
            sample = _sample(complete, max_points, seed + index)
            fig.add_trace(
                go.Scatter(
                    x=sample[:, 0].tolist(),
                    y=sample[:, 1].tolist(),
                    mode="markers",
                    marker={"color": COLORS[role], "size": 5, "opacity": 0.5},
                    name=label,
                    showlegend=False,
                ),
                row=1,
                col=index + 1,
            )
            meta["sources"][role] = {
                "rows": len(frame),
                "available": len(complete),
                "plotted": len(sample),
            }
            fig.update_xaxes(title_text=text(column_1), row=1, col=index + 1)
            fig.update_yaxes(title_text=text(column_2), row=1, col=index + 1)
        # Matching coordinate ranges make differences interpretable across panels.
        fig.update_xaxes(matches="x")
        fig.update_yaxes(matches="y")
    elif len(numeric) == 1:
        number = numeric[0]
        category = column_2 if number == column_1 else column_1
        codes, labels = _categorical(frames[0][category], frames[1][category], max_categories)
        fig = go.Figure()
        for index, (frame, array, role, label) in enumerate(
            zip(frames, codes, ("reference", "generated"), ("Real", "Generated"), strict=True)
        ):
            values = numeric_values(frame[number])
            valid = (array >= 0) & np.isfinite(values)
            positions = _sample(np.flatnonzero(valid), max_points, seed + index)
            fig.add_trace(
                go.Box(
                    x=[labels[array[i]] for i in positions],
                    y=values[positions].tolist(),
                    name=label,
                    marker_color=COLORS[role],
                    boxpoints=False,
                )
            )
            meta["sources"][role] = {
                "rows": len(frame),
                "available": int(valid.sum()),
                "plotted": len(positions),
            }
        fig.update_layout(boxmode="group")
        fig.update_xaxes(
            title_text=text(category), type="category", categoryorder="array", categoryarray=labels
        )
        fig.update_yaxes(title_text=text(number))
    else:
        xcodes, xlabels = _categorical(frames[0][column_1], frames[1][column_1], max_categories)
        ycodes, ylabels = _categorical(frames[0][column_2], frames[1][column_2], max_categories)
        matrices, counts = [], []
        for x, y in zip(xcodes, ycodes, strict=True):
            valid = (x >= 0) & (y >= 0)
            count = np.bincount(
                y[valid] * len(xlabels) + x[valid], minlength=len(xlabels) * len(ylabels)
            ).reshape(len(ylabels), len(xlabels))
            matrices.append(count / valid.sum() if valid.any() else np.full(count.shape, np.nan))
            counts.append(int(valid.sum()))
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Real", "Generated"])
        finite = [matrix[np.isfinite(matrix)] for matrix in matrices]
        vmax = max([float(values.max()) for values in finite if values.size], default=1.0)
        for index, (matrix, role, count, frame) in enumerate(
            zip(matrices, ("reference", "generated"), counts, frames, strict=True)
        ):
            fig.add_trace(
                go.Heatmap(
                    z=matrix.tolist(),
                    x=xlabels,
                    y=ylabels,
                    coloraxis="coloraxis",
                    hoverongaps=False,
                    hovertemplate="%{x} × %{y}<br>Joint fraction: %{z:.4f}<extra></extra>",
                ),
                row=1,
                col=index + 1,
            )
            meta["sources"][role] = {"rows": len(frame), "available": count, "plotted": count}
            fig.update_xaxes(title_text=text(column_1), type="category", row=1, col=index + 1)
            fig.update_yaxes(title_text=text(column_2), type="category", row=1, col=index + 1)
        fig.update_layout(
            coloraxis={
                "cmin": 0,
                "cmax": vmax,
                "colorscale": "Viridis",
                "colorbar": {"title": "Joint fraction"},
            }
        )
    return finish(fig, text(f"{column_1} × {column_2} · real and generated"), metadata=meta)
