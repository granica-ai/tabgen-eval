"""Render landing-page evidence from CPU examples and recorded benchmarks."""

from __future__ import annotations

import json
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean

import pandas as pd
import plotly.graph_objects as go
from figure_assets import write_figure

from tabgen_eval import EvalData, Schema, evaluate


def _reference_figure(report):
    """Overlay repeated-score means and raw percentile bands on one row."""
    figure = report.fidelity.plot("scores", metrics=["shape"], summary="mean")
    # The page supplies its own title for this single metric panel.
    figure.layout.annotations = tuple(
        annotation for annotation in figure.layout.annotations if annotation.yref != "paper"
    )
    score = report["shape/shape"]
    records, endpoints = [], []
    for label, distribution in (
        ("synthetic vs. real", score.generated_distribution),
        ("real vs. real", score.reference_distribution),
    ):
        if distribution is None or not distribution.finite_count:
            raise ValueError(f"Homepage Shape comparison requires finite {label} repetitions")
        point = distribution.mean
        low, high = distribution.band_low, distribution.band_high
        endpoints.extend((point, low, high))
        records.append(
            {
                "Comparison": label,
                "Mean Shape score": round(point, 4),
                "Score band low": round(low, 4),
                "Score band high": round(high, 4),
                "Rows per role": score.n_eff,
                "Repeated comparisons": distribution.finite_count,
            }
        )
    figure.update_xaxes(
        range=[max(0, min(endpoints) - 0.05), min(1, max(endpoints) + 0.03)],
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#dce3e8",
        ticks="outside",
        tickcolor="#dce3e8",
        ticklen=5,
    )
    figure.update_yaxes(visible=False, range=[-0.5, 0.5], fixedrange=True)
    figure.update_layout(
        template="plotly_white",
        title={"text": "Example: Shape scores and uncertainty", "x": 0.03},
        height=270,
        margin={"l": 35, "r": 30, "t": 65, "b": 55},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26323d"},
        showlegend=False,
    )
    return figure, pd.DataFrame(records)


def _latency_figure(benchmark):
    """Show the recorded overall means on a zero-based linear axis."""
    datasets = [row["dataset"] for row in benchmark["datasets"]]
    sources = benchmark["sample_sources"]
    groups = benchmark["computation_groups"]
    configurations = benchmark["configurations"]
    expected = set(product(datasets, sources, groups, configurations))
    seen, timings = set(), {key: [] for key in configurations}
    for row in benchmark["records"]:
        identity = tuple(row[key] for key in ("dataset", "sample", "computation", "configuration"))
        if identity not in expected or identity in seen:
            raise ValueError(f"Unexpected or duplicate homepage timing: {identity}")
        seconds = row["seconds"]
        if not isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid homepage timing: {identity}")
        seen.add(identity)
        timings[row["configuration"]].append(seconds)
    if seen != expected or len(expected) != 660:
        raise ValueError("Homepage latency requires the complete 660-call benchmark")
    labels = list(configurations.values())
    values = [mean(timings[key]) for key in configurations]
    baseline = mean(timings["python_cpu"])
    bar_labels = [
        f"{value:.2f} s"
        if key == "python_cpu"
        else f"<b>{baseline / value:.1f}x faster</b><br>{value:.2f} s"
        for key, value in zip(configurations, values, strict=False)
    ]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#a3adb7", "#188477", "#d27622"],
            text=bar_labels,
            textposition="outside",
            textfont={"size": 14},
            constraintext="none",
            cliponaxis=False,
            hovertemplate="%{x}<br>Mean latency: %{y:.3f} seconds<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        showlegend=False,
        bargap=0.65,
        uniformtext={"minsize": 14, "mode": "show"},
        title={"text": "Average seconds per evaluation", "x": 0.03},
        height=340,
        margin={"l": 70, "r": 25, "t": 70, "b": 60},
        font={"family": "Arial, sans-serif", "size": 14, "color": "#26323d"},
        yaxis={
            "type": "linear",
            "range": [0, max(values) * 1.18],
            "visible": True,
            "showline": True,
            "linecolor": "#a3adb7",
            "linewidth": 1,
            "ticks": "outside",
            "showgrid": False,
            "zeroline": False,
        },
        xaxis={
            "fixedrange": True,
            "showline": True,
            "linecolor": "#a3adb7",
            "linewidth": 1,
            "showgrid": False,
            "zeroline": False,
            "automargin": True,
            "tickmode": "array",
            "tickvals": labels,
            "ticktext": [
                {"Optimized CPU": "TabGenEval-CPU", "GPU": "TabGenEval-GPU"}.get(label, label)
                for label in labels
            ],
        },
    )
    table = pd.DataFrame(
        {
            "Execution": labels,
            "Mean seconds per computation": [round(value, 3) for value in values],
            "Completed calls": [len(timings[key]) for key in configurations],
        }
    )
    return figure, table


def _speedup_figure(latency_figure):
    """Show the ratio of mean times relative to the Python CPU baseline."""
    figure = go.Figure(latency_figure)
    bars = figure.data[0]
    baseline = bars.y[list(bars.x).index("Python CPU")]
    seconds = list(bars.y)
    speedups = [baseline / seconds for seconds in bars.y]
    bars.y = speedups
    bars.text = [
        f"<b>{'1x' if value == 1 else f'{value:.1f}x'}</b><br>{elapsed:.2f} s"
        for value, elapsed in zip(speedups, seconds, strict=False)
    ]
    bars.hovertemplate = "%{x}<br>Speedup vs. Python CPU: %{y:.1f}x<extra></extra>"
    figure.update_layout(title_text="Speedup relative to Python CPU")
    figure.update_yaxes(range=[0, max(speedups) * 1.18], ticksuffix="x")
    table = pd.DataFrame({"Execution": list(bars.x), "Speedup vs. Python CPU": speedups})
    return figure, table


def generate_landing_examples(app):
    """Reuse the walkthrough data and compute uncertainty on both sides."""
    example, perturbed, holdout = app._ltm_visualization_example
    _, generated = example["make_data"]()
    comparison = evaluate(
        EvalData(test=holdout, generated=generated, schema=Schema(**perturbed.schema_dict)),
        metrics=["shape"],
        device="cpu",
        seed=23,
        uncertainty={"mode": "both", "n_replicates": 10},
    )
    output = Path(app.srcdir) / "_generated" / "landing"
    output.mkdir(parents=True, exist_ok=True)
    figure, reference_table = _reference_figure(comparison)
    write_figure(
        output,
        "reference",
        figure,
        "Generated-real and real-real Shape scores with 95% score bands",
        reference_table,
        figure_id="ltm-home-reference",
        show_values=False,
    )
    figure = perturbed.fidelity.plot("columns", metric="shape")
    write_figure(
        output,
        "columns",
        figure,
        "Shape scores for the perturbed sample's columns",
        perturbed.details("shape").round(4),
        figure_id="ltm-home-columns",
    )
    latency_path = Path(app.confdir).parent / "assets" / "homepage-latency.json"
    latency = json.loads(latency_path.read_text())
    figure, table = _latency_figure(latency)
    write_figure(
        output,
        "latency",
        figure,
        "Mean evaluation latency: Python CPU 233.35 seconds, optimized CPU 6.99 seconds, GPU 2.57 seconds",
        table,
        figure_id="ltm-home-latency",
        static_plot=False,
        show_values=False,
    )
    (output / "latency-values.html").write_text(table.to_html(index=False, border=0) + "\n")
    speedup_figure, speedup_table = _speedup_figure(figure)
    write_figure(
        output,
        "speedup",
        speedup_figure,
        "Speedup relative to Python CPU: Python CPU 1x, optimized CPU 33.4x, GPU 90.8x",
        speedup_table,
        figure_id="ltm-home-speedup",
        show_values=False,
    )
    datasets = pd.DataFrame(
        [
            {
                "Dataset": row["dataset"].replace("_", " ").title(),
                "Training rows": f"{row['train_rows']:,}",
                "Test / sample rows": f"{row['test_and_generated_rows']:,}",
                "Columns": row["columns"],
            }
            for row in latency["datasets"]
        ]
    )
    (output / "datasets.html").write_text(
        '<div class="ltm-home-datasets" role="region" tabindex="0" '
        'aria-label="Benchmark datasets and input file sizes">'
        + datasets.to_html(index=False, border=0, classes="ltm-home-dataset-table")
        + "</div>\n"
    )
    (output / "reference-values.json").write_text(
        json.dumps({"comparisons": reference_table.to_dict("records")}, indent=2) + "\n"
    )


def add_landing_style(app, pagename, templatename, context, doctree):
    if pagename == "index":
        app.add_css_file("landing.css")


def setup(app):
    app.connect("builder-inited", generate_landing_examples, priority=700)
    app.connect("html-page-context", add_landing_style)
