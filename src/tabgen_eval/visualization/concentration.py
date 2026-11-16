"""Mean and upper local-mass profiles at actually evaluated radii."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ._data import detail_table, row_usage_hover
from ._style import COLORS, finish, panel_grid

if TYPE_CHECKING:
    from tabgen_eval.report import Report


def concentration(report: Report) -> go.Figure:
    """Compare the captured real and generated local-mass profiles.

    Radii are shared within this fixed comparison. Lines connect evaluated
    radii for readability; they do not evaluate any intermediate radius.
    """
    frame = detail_table(report, "concentration", "radius").sort_values("radius", kind="stable")
    result = next(result for result in report.results if result.name == "concentration")
    usage_hover = row_usage_hover(result)
    quantile = frame.attrs.get("metadata", {}).get("upper_quantile", 0.95)
    upper = frame.loc[frame["radius_id"] != "radii_1e-3"]
    summaries = [("mean", frame)]
    titles = ["Mean local mass"]
    if not upper.empty:
        summaries.append(("upper", upper))
        titles.append(f"Local mass quantile {quantile:g}")
    fig, layout = panel_grid(titles, panel_height=220)
    for col, (summary, selected) in enumerate(summaries, 1):
        for role, prefix, label in (
            ("reference", "reference", "Real"),
            ("generated", "candidate", "Generated"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=selected["radius"].tolist(),
                    y=selected[f"{prefix}_{summary}"].tolist(),
                    customdata=selected[["radius_id", f"{prefix}_n"]].to_numpy().tolist(),
                    mode="lines+markers",
                    name=label,
                    legendgroup=role,
                    showlegend=col == 1,
                    line={"color": COLORS[role]},
                    hovertemplate=(
                        "Radius: %{x:.5g} (%{customdata[0]})<br>Local mass: %{y:.5g}"
                        "<br>Complete selected rows: %{customdata[1]}"
                        + ("<br>" + usage_hover if usage_hover else "")
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Evaluated radius", row=1, col=col)
        fig.update_yaxes(title_text="Fraction of other records", row=1, col=col)
    return finish(
        fig,
        "Concentration at shared radii",
        **layout,
        metadata={
            "kind": "concentration",
            "upper_quantile": quantile,
            "uncertainty": "none; descriptive point profiles",
        },
    )
