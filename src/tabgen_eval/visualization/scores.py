"""Scalar comparison figures with one independent scale per statistic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go

from tabgen_eval.metrics.base import Direction
from tabgen_eval.report.roles import reference_only, score_role_label

from ._data import metric_results, row_usage_hover, stored_band
from ._style import COLORS, finish, panel_grid, panel_title, status, wrapped_text

if TYPE_CHECKING:
    from tabgen_eval.report import Report


def scores(
    report: Report,
    *,
    metrics: str | Sequence[str] | None = None,
    summary: Literal["point", "mean"] = "point",
) -> go.Figure:
    """Plot stored score estimates and requested raw percentile bands.

    Each statistic has its own horizontal scale, including separate PRDC
    statistics. Bands belong to retained score distributions and are drawn
    without shifting their endpoints to center them on the displayed estimate.
    Repeated evaluation already stores means as its estimates; single comparisons
    remain point scores.
    Set ``summary="mean"`` to show each retained distribution's mean instead;
    roles without finite repetitions retain their stored point or absent status.
    """
    if summary not in {"point", "mean"}:
        raise ValueError("summary must be 'point' or 'mean'")
    entries = [
        (result, stat)
        for result in metric_results(report, metrics)
        for stat in result.stats.values()
    ]
    if not entries:
        raise ValueError("The selected report contains no statistics")
    cols = min(2, len(entries))
    titles = []
    for _, stat in entries:
        if stat.direction is Direction.TARGET:
            direction = (
                f"target {stat.target:g}" if stat.target is not None else "target comparison"
            )
        else:
            direction = {
                Direction.HIGHER_BETTER: "higher is better",
                Direction.LOWER_BETTER: "lower is better",
            }[stat.direction]
        titles.append(panel_title(stat.display, direction))
    fig, layout = panel_grid(titles, columns=cols, panel_height=150)
    layout["top"] -= 40  # No overall title above the metric headings.
    for index, (result, stat) in enumerate(entries):
        row, col = index // cols + 1, index % cols + 1
        usage_hover = row_usage_hover(result)
        plotted_values = []
        generated_label = (
            "Generated-trained"
            if result.name == "mle" and stat.name == "score"
            else "synthetic vs. real"
        )
        reference_label = "Real-trained" if result.name == "mle" else "real vs. real"
        if reference_only(stat):
            label = score_role_label(stat)
            roles = [
                ("reference", label, stat.value, stat.reference_distribution, stat.point_status)
            ]
        else:
            if stat.score_role != "generated_test":
                generated_label = score_role_label(stat)
            primary_color = (
                "reference" if stat.score_role in {"test_train", "train_generated"} else "generated"
            )
            roles = [
                (
                    primary_color,
                    generated_label,
                    stat.value,
                    stat.generated_distribution,
                    stat.point_status,
                )
            ]
            if not np.isnan(stat.ideal) or stat.reference_distribution is not None:
                roles.append(
                    (
                        "reference",
                        reference_label,
                        stat.ideal,
                        stat.reference_distribution,
                        stat.ideal_status,
                    )
                )
        displayed_roles = []
        for role, label, point, distribution, point_status in roles:
            estimate = (
                stat.ideal_estimate
                if label == reference_label and not reference_only(stat)
                else stat.value_estimate
            )
            if summary == "mean" and distribution is not None and distribution.finite_count:
                point, point_status, estimate = distribution.mean, None, "mean"
            displayed_roles.append((role, label, point, distribution, point_status, estimate))
        finite_points = [item[2] for item in displayed_roles if np.isfinite(item[2])]
        extent = [*finite_points]
        for item in displayed_roles:
            band = stored_band(item[3])
            if band is not None:
                extent.extend(band)
        span = max(extent) - min(extent) if extent else 0
        close_labels = len(finite_points) > 1 and abs(finite_points[0] - finite_points[1]) <= max(
            span * (0.5 if cols > 1 else 0.25), 1e-12
        )
        for role_index, (role, label, point, distribution, point_status, estimate) in enumerate(
            displayed_roles
        ):
            band = stored_band(distribution)
            if band is not None:
                plotted_values.extend(band)
                fig.add_trace(
                    go.Scatter(
                        x=list(band),
                        y=[0, 0],
                        mode="lines+markers",
                        line={"color": COLORS[role], "width": 14},
                        marker={"color": COLORS[role], "symbol": "circle", "size": 14},
                        opacity=0.22,
                        name=f"{label} score band",
                        legendgroup=label,
                        showlegend=False,
                        customdata=[
                            [
                                stat.n_eff,
                                distribution.finite_count,
                                distribution.requested_count,
                                usage_hover,
                            ]
                        ]
                        * 2,
                        hovertemplate=(
                            f"{label} {100 * distribution.confidence_interval:g}% "
                            f"score band: [{band[0]:.5g}, {band[1]:.5g}]"
                            "<br>Selected rows per role: %{customdata[0]}"
                            "<br>Finite score replicates: %{customdata[1]}/%{customdata[2]}"
                            + ("<br>%{customdata[3]}" if usage_hover else "")
                            + "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            if point_status == "absent":
                if band is not None:
                    fig.add_annotation(
                        x=(band[0] + band[1]) / 2,
                        y=0,
                        yshift=34,
                        text=f"<b>{wrapped_text(label, width=18)}</b>",
                        showarrow=False,
                        font={"color": COLORS[role], "size": 14},
                        row=row,
                        col=col,
                    )
                elif distribution is not None and not distribution.finite_count:
                    fig.add_annotation(
                        text=f"{label}: {distribution.reason}",
                        x=0.5,
                        y=0.5,
                        xref="x domain",
                        yref="y domain",
                        showarrow=False,
                        row=row,
                        col=col,
                    )
                continue
            if np.isfinite(point):
                plotted_values.append(point)
            fig.add_trace(
                go.Scatter(
                    x=[float(point) if np.isfinite(point) else None],
                    y=[0],
                    mode="markers",
                    marker={
                        "color": COLORS[role],
                        "size": 16,
                        "line": {"color": "white", "width": 2.5},
                    },
                    name=f"{label} {estimate}",
                    legendgroup=label,
                    showlegend=False,
                    customdata=[[stat.n_eff, usage_hover]],
                    hovertemplate=(
                        f"{label} {estimate}: %{{x:.6g}}<br>Selected rows per role: %{{customdata[0]}}"
                        + ("<br>%{customdata[1]}" if usage_hover else "")
                        + "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            if np.isfinite(point):
                offset = role_index if close_labels else 0
                fig.add_annotation(
                    x=point,
                    y=0,
                    yshift=34 + offset * 28,
                    text=f"<b>{wrapped_text(label, width=18)}</b>",
                    showarrow=False,
                    font={"color": COLORS[role], "size": 14},
                    row=row,
                    col=col,
                )
                fig.add_annotation(
                    x=point,
                    y=0,
                    yshift=-44,
                    text=f"{point:.3f}",
                    showarrow=False,
                    font={"color": "#52616d", "size": 14},
                    row=row,
                    col=col,
                )
            if not np.isfinite(point):
                fig.add_annotation(
                    text=f"{label}: {status(point)}",
                    x=0.5,
                    y=0.8 if role == "generated" else 0.2,
                    xref="x domain",
                    yref="y domain",
                    showarrow=False,
                    row=row,
                    col=col,
                )
        yaxis = fig.layout["yaxis" if index == 0 else f"yaxis{index + 1}"]
        fig.update_xaxes(
            anchor="free",
            position=sum(yaxis.domain) / 2,
            row=row,
            col=col,
        )
        if plotted_values and min(plotted_values) == max(plotted_values):
            # A lone large value otherwise gets a microscopic automatic range
            # and nearly identical, very long tick labels. Retained bands and
            # comparisons with different values keep their automatic ranges.
            fig.update_xaxes(rangemode="tozero", row=row, col=col)
        fig.update_yaxes(
            visible=False,
            range=[-0.5, 0.5],
            fixedrange=True,
            row=row,
            col=col,
        )
    fig = finish(
        fig,
        "",
        **layout,
        metadata={
            "kind": "scores",
            "bands": "stored raw percentile score bands",
            "summary": summary,
        },
    )
    fig.update_layout(
        height=fig.layout.height - 40,
        margin={"l": 35, "r": 30, "b": 25},
    )
    return fig
