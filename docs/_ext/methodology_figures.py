"""Render Gaussian examples for the statistical methodology guide."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from figure_assets import write_figure
from plotly.subplots import make_subplots

COLORS = ("#287c8e", "#c46532", "#6972ae")


def _style(figure, title, height):
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.03},
        height=height,
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26323d"},
        margin={"l": 85, "r": 30, "t": 100, "b": 110},
        legend={"orientation": "h", "y": -55 / max(1, height - 210), "yanchor": "top", "x": 0},
    )
    figure.update_annotations(font_size=14)
    return figure


def mean_estimates_figure(data):
    """Compare saved score distributions with their known finite-sample means."""
    from tabgen_eval.metrics import Shape, Trend
    from tabgen_eval.report import Report
    from tabgen_eval.report.assembly import assemble_stat
    from tabgen_eval.report.result import MetricResult

    sample = data["resampling"]
    results = []
    for metric in (Shape, Trend):
        spec = metric.stats[0]
        stat = assemble_stat(
            spec=spec,
            value_reps=sample["scores"]["generated"][metric.name],
            ideal_reps=sample["scores"]["reference"][metric.name],
            value_point=float("nan"),
            ideal_point=float("nan"),
            base_flags=(),
            insufficient=False,
            not_applicable=False,
            n_holdout=sample["holdout_rows"],
            n_generated=sample["generated_rows"],
            n_eff=sample["rows_per_sample"],
        )
        results.append(
            MetricResult(
                metric.name,
                metric.category,
                stats={spec.name: stat},
                primary=spec.name,
            )
        )
    report = Report(
        results=results,
        schema_dict={"numeric": data["population"]["columns"]},
        options_dict={"uncertainty": {"mode": "both", "n_replicates": sample["n_replicates"]}},
    )
    figure = report.fidelity.plot(kind="scores", metrics=["shape", "trend"])
    roles = (("reference", "Real–real"), ("generated", "Generated–real"))
    records = []
    for col, metric in enumerate(("shape", "trend"), 1):
        known_means = {}
        extents = []
        for role, label in roles:
            mean = sample["mean"][role][metric]
            exact = sample["exact_mean"][role][metric]
            lower = sample["band"][role][metric]["low"]
            upper = sample["band"][role][metric]["high"]
            known_means.setdefault(exact, []).append(label)
            extents.extend((lower, upper, mean, exact))
            records.append(
                {
                    "Metric": metric.title(),
                    "Comparison": label,
                    "Exact mean": exact,
                    "Repeated-score mean": mean,
                    "Band low": lower,
                    "Band high": upper,
                }
            )
        figure.add_trace(
            go.Scatter(
                x=list(known_means),
                y=[0] * len(known_means),
                mode="markers",
                marker={
                    "color": "#77828c",
                    "symbol": "line-ns",
                    "size": 24,
                    "line": {"color": "#77828c", "width": 2},
                },
                name="Known mean",
                showlegend=False,
                customdata=[" / ".join(labels) for labels in known_means.values()],
                hovertemplate="%{customdata}<br>Known mean: %{x:.9f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        padding = max((max(extents) - min(extents)) * 0.08, 0.0003)
        figure.update_xaxes(range=[min(extents) - padding, max(extents) + padding], row=1, col=col)
    return figure, pd.DataFrame(records).round(6)


def target_figure(frame):
    """Compare single realized scores with the exact finite-sample expectations."""
    frame = frame.sort_values("n")
    selected = frame.loc[frame.highlight]
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=["Shape", "Trend"], horizontal_spacing=0.16
    )
    for col, metric in enumerate(("shape", "trend"), 1):
        figure.add_trace(
            go.Scatter(
                x=frame.n.tolist(),
                y=frame[metric].tolist(),
                mode="lines+markers",
                line={"color": COLORS[0], "width": 2, "dash": "solid"},
                marker={"color": COLORS[0], "size": 6},
                name="<i>μ</i><sub>n</sub>(<i>P</i>)",
                legendgroup="exact",
                legendrank=1,
                showlegend=col == 1,
                hovertemplate="n=%{x:,}<br>Exact expectation: %{y:.9f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=[int(frame.n.min()), int(frame.n.max())],
                y=[1, 1],
                mode="lines",
                line={"color": "#77828c", "dash": "dash"},
                name="Population ideal",
                legendgroup="limit",
                legendrank=2,
                showlegend=col == 1,
                hovertemplate="Population ideal: 1<extra></extra>",
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=frame.n.tolist(),
                y=frame[f"{metric}_score"].tolist(),
                mode="lines+markers",
                line={"color": COLORS[1], "width": 2, "dash": "solid"},
                marker={"color": COLORS[1], "size": 6},
                name="<i>T</i>(<i>R</i><sub>n</sub>, <i>R</i>′<sub>n</sub>)",
                legendgroup="realized",
                legendrank=0,
                showlegend=col == 1,
                hovertemplate="n=%{x:,}<br>Realized score: %{y:.9f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        figure.update_xaxes(
            type="log",
            tickvals=selected.n.tolist(),
            title_text="Rows per sample n",
            row=1,
            col=col,
        )
        figure.update_yaxes(
            title_text="Real–real score",
            range=[
                float(frame[["shape", "trend", "shape_score", "trend_score"]].min().min()) - 0.002,
                1.002,
            ],
            row=1,
            col=col,
        )
    return _style(figure, "Finite samples have their own performance target", 430).update_layout(
        legend_traceorder="normal"
    )


def generate_methodology_figures(app):
    root = Path(app.confdir)
    data = json.loads((root.parent / "assets/methodology-gaussian.json").read_text())
    frame = pd.DataFrame(data["targets"])
    output = root / "_generated/methodology"
    output.mkdir(parents=True, exist_ok=True)
    entries = [
        (
            "target",
            target_figure(frame),
            frame[["n", "shape", "shape_score", "trend", "trend_score"]]
            .rename(
                columns={
                    "n": "Rows per sample",
                    "shape": "Exact Shape mean",
                    "shape_score": "Realized Shape score",
                    "trend": "Exact Trend mean",
                    "trend_score": "Realized Trend score",
                }
            )
            .round(9),
        ),
    ]
    figure, table = mean_estimates_figure(data)
    entries.append(("mean-estimates", figure, table))
    for name, figure, table in entries:
        write_figure(
            output,
            name,
            figure,
            f"Methodology: {name}",
            table,
            figure_id=f"ltm-method-{name}",
            table_caption="Repeated-score means and descriptive percentile bands"
            if name == "mean-estimates"
            else None,
        )


def add_style(app, pagename, templatename, context, doctree):
    if pagename == "methodology":
        app.add_css_file("methodology.css")


def setup(app):
    app.connect("builder-inited", generate_methodology_figures, priority=750)
    app.connect("html-page-context", add_style)
