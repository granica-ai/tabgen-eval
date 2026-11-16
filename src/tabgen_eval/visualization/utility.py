"""Raw predictive performance and generated-to-real score ratios."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from tabgen_eval.metrics.base import Direction

from ._data import metric_results, row_usage_hover
from ._style import COLORS, finish, panel_grid, panel_title, status, wrapped_text

if TYPE_CHECKING:
    from tabgen_eval.report import Report
    from tabgen_eval.report.result import MetricResult


def _score_labels(scoring: str) -> tuple[str, str]:
    """Present the task's raw score and its corresponding ratio direction."""
    if scoring == "mae":
        return "MAE", "lower is better"
    if scoring == "auroc":
        return "AUROC", "higher is better"
    return scoring, ""


def _prediction_summary(result: MetricResult, item: dict[str, object]) -> tuple[str, float] | None:
    for name, label in (
        ("prediction_agreement", "Prediction agreement"),
        ("prediction_mse", "Prediction MSE"),
    ):
        stat = result.stats.get(name)
        if stat is not None and "NOT_APPLICABLE" in stat.flag_codes:
            continue
        if name in item:
            value = item[name]
            if not isinstance(value, Real):
                raise TypeError(f"Stored {name} must be a real numeric score")
            return label, float(value)
        if stat is not None:
            return label, stat.value
    return None


def utility(report: Report, *, metrics: str | Sequence[str] | None = None) -> go.Figure:
    """Plot reported predictive estimates, with legacy diagnostic-only support.

    Each target receives independent score and ratio axes. Classification AUROC
    and its ratio are higher-is-better; regression MAE and its ratio are
    lower-is-better. Nonfinite ratios are labelled explicitly. No models are
    fitted and no scores are recomputed.
    """
    if report.provenance.get("evaluation_kind") == "test_only":
        from .scores import scores

        selected = [
            result.name for result in metric_results(report, metrics) if result.name == "mle"
        ]
        if not selected:
            raise ValueError("No stored raw predictive scores available; select MLE")
        return scores(report, metrics=selected)
    records = []
    selected_results = metric_results(report, metrics)
    usage = {result.name: row_usage_hover(result) for result in selected_results}
    for result in selected_results:
        if result.name != "mle":
            continue
        info = result.diagnostics
        score = result.stats.get("score")
        if score is not None:
            # Repeated reports have no separate point outcome. Read all estimates
            # from their statistics, including the mean of the per-draw ratios.
            diagnostic = (info.get("point") or {}).get("diagnostics", {})
            if not diagnostic:
                diagnostic = next(
                    (
                        item["diagnostics"]
                        for item in info.get("replicates", [])
                        if item.get("diagnostics", {}).get("scoring")
                    ),
                    {},
                )
            target = diagnostic.get("target", report.schema_dict.get("target") or "target")
            item = {
                "generated_score": score.value,
                "reference_score": score.ideal,
                "ratio": result.stats["ratio"].value if "ratio" in result.stats else np.nan,
                "scoring": diagnostic.get(
                    "scoring", "mae" if score.direction is Direction.LOWER_BETTER else "auroc"
                ),
            }
            items = [(target, item)]
        elif "point" in info:
            point = info["point"]
            items = [(point.get("diagnostics", {}).get("target", "target"), point)]
        else:
            items = [(info.get("target", "target"), info)]
        for target, item in items:
            if "generated_score" in item and "reference_score" in item:
                scoring = item.get(
                    "scoring", item.get("diagnostics", {}).get("scoring", "predictive score")
                )
                records.append(
                    (result.name, target, scoring, item, _prediction_summary(result, item))
                )
    if not records:
        raise ValueError("No stored raw predictive scores available; select MLE")
    show_predictions = any(prediction is not None for *_, prediction in records)
    titles = []
    for metric, target, scoring, _, prediction in records:
        score_label, direction = _score_labels(scoring)
        titles.extend(
            [
                panel_title(f"{metric} · {target}", f"Raw {score_label} · {direction}"),
                panel_title(f"{score_label} ratio", f"{direction} · 1 = equal performance"),
            ]
        )
        if show_predictions:
            titles.extend(
                [
                    panel_title(
                        prediction[0] if prediction is not None else "Shared-test predictions"
                    ),
                    "",
                ]
            )
    fig, layout = panel_grid(titles)
    for index, (metric, _, scoring, item, prediction) in enumerate(records):
        row = index * (2 if show_predictions else 1) + 1
        score_label, direction = _score_labels(scoring)
        source_hover = "<br>" + usage[metric] if usage[metric] else ""
        score_hover = (
            f"Raw {score_label}: "
            + "%{x:.6g}"
            + ("<br>" + direction if direction else "")
            + source_hover
            + "<extra></extra>"
        )
        for role, key, label in (
            ("generated", "generated_score", "Generated-trained"),
            ("reference", "reference_score", "Real-trained"),
        ):
            value = item[key]
            fig.add_trace(
                go.Scatter(
                    x=[value if np.isfinite(value) else None],
                    y=[label],
                    mode="markers",
                    marker={"color": COLORS[role], "size": 11},
                    name=label,
                    legendgroup=role,
                    showlegend=row == 1,
                    hovertemplate=score_hover,
                ),
                row=row,
                col=1,
            )
            if not np.isfinite(value):
                fig.add_annotation(
                    text=f"{label}: {status(value)}",
                    x=0.5,
                    y=0.75 if role == "generated" else 0.25,
                    xref="x domain",
                    yref="y domain",
                    showarrow=False,
                    row=row,
                    col=1,
                )
        ratio = item.get("ratio", float("nan"))
        fig.add_trace(
            go.Scatter(
                x=[ratio if np.isfinite(ratio) else None],
                y=["Ratio"],
                mode="markers",
                marker={"color": COLORS["generated"], "size": 11},
                showlegend=False,
                hovertemplate=(
                    f"Generated / real {score_label}: "
                    + "%{x:.6g}"
                    + ("<br>" + direction if direction else "")
                    + "<br>1 = equal raw performance"
                    + source_hover
                    + "<extra></extra>"
                ),
            ),
            row=row,
            col=2,
        )
        if not np.isfinite(ratio):
            fig.add_annotation(
                text=f"Ratio: {status(ratio)}",
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                row=row,
                col=2,
            )
        fig.update_xaxes(title_text=score_label, rangemode="tozero", row=row, col=1)
        fig.update_xaxes(
            title_text=wrapped_text(f"Generated / real {score_label}"),
            rangemode="tozero",
            row=row,
            col=2,
        )
        labels = ["Generated-trained", "Real-trained"]
        fig.update_yaxes(
            type="category",
            tickmode="array",
            tickvals=labels,
            ticktext=[wrapped_text(label, width=18) for label in labels],
            row=row,
            col=1,
        )
        fig.update_yaxes(type="category", row=row, col=2)
        if show_predictions and prediction is not None:
            label, value = prediction
            fig.add_trace(
                go.Scatter(
                    x=[value if np.isfinite(value) else None],
                    y=[label],
                    mode="markers",
                    marker={"color": COLORS["generated"], "size": 11},
                    showlegend=False,
                    hovertemplate=label + ": %{x:.6g}" + source_hover + "<extra></extra>",
                ),
                row=row + 1,
                col=1,
            )
            if not np.isfinite(value):
                fig.add_annotation(
                    text=f"{label}: {status(value)}",
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    showarrow=False,
                    row=row + 1,
                    col=1,
                )
            fig.update_xaxes(title_text="Score", rangemode="tozero", row=row + 1, col=1)
            fig.update_yaxes(
                type="category",
                tickmode="array",
                tickvals=[label],
                ticktext=[wrapped_text(label, width=18)],
                row=row + 1,
                col=1,
            )
    return finish(
        fig,
        "Predictive utility",
        **layout,
        metadata={"kind": "utility", "uncertainty": "stored predictive score estimates"},
    )
