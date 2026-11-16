"""Build interactive report figures from a small, reproducible CPU evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tabgen_eval import EvalData, Schema, evaluate
from tabgen_eval.visualization import plot_column_pair


def make_data(*, perturb=True):
    """Return mixed-type samples, optionally adding three generator defects."""
    rng = np.random.default_rng(23)

    def sample(size):
        visits = rng.normal(12, 3, size)
        return pd.DataFrame(
            {
                "visits": visits,
                "spend": 20 * visits + rng.normal(0, 15, size),
                "tenure": rng.uniform(1, 12, size),
                "segment": rng.choice(
                    ["local", "regional", "online"],
                    size,
                    p=[0.55, 0.30, 0.15],
                ),
            }
        )

    holdout, generated = sample(600), sample(300)
    if perturb:
        generated["tenure"] += 5
        generated["spend"] = rng.permutation(generated["spend"].to_numpy())
        generated.iloc[:75] = generated.iloc[150].to_numpy()
    return holdout, generated


def make_report(holdout, generated):
    """Evaluate once, retaining the component scores used by the figures."""
    return evaluate(
        EvalData(
            test=holdout,
            generated=generated,
            schema=Schema(numeric=["visits", "spend", "tenure"], categorical=["segment"]),
        ),
        metrics=[
            "shape",
            "trend",
            "cond_shape",
            "cond_trend",
            "recovery",
            "precision",
            "recall",
            "density",
            "coverage",
            "concentration",
        ],
        details=True,
        device="cpu",
        seed=23,
        uncertainty={"mode": "reference", "n_replicates": 10},
    )


def make_figures(report, holdout, generated):
    """Plot stored scores and one bounded view of the original input tables."""
    fidelity = report.fidelity
    figures = {
        "scores": fidelity.plot(kind="scores", metrics=["shape", "trend"]),
        "columns": fidelity.plot(kind="columns", metric="shape"),
        "pairs": fidelity.plot(kind="pairs", metric="trend"),
        "slices": fidelity.plot(kind="slices", metric="cond_shape", column="segment", stat="shape"),
        "concentration": fidelity.plot(kind="concentration"),
        "prdc": fidelity.plot(
            kind="scores", metrics=["precision", "recall", "density", "coverage"]
        ),
        "association": plot_column_pair(
            holdout,
            generated,
            "visits",
            "spend",
            schema=Schema(numeric=["visits", "spend"]),
        ),
    }
    for figure in figures.values():
        figure.update_layout(font={"family": "Arial, sans-serif"})
    return figures


def write_demo(output):
    """Write standalone HTML figures sharing one local Plotly JavaScript file."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    holdout, generated = make_data()
    report = make_report(holdout, generated)
    for name, figure in make_figures(report, holdout, generated).items():
        figure.write_html(
            output / f"{name}.html",
            include_plotlyjs="directory",
            config={"responsive": True, "displaylogo": False},
        )
    report.to_json(str(output / "report.json"))
    print(report.summary())
    print(f"Interactive figures: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("visualization-demo"))
    write_demo(parser.parse_args().output)
