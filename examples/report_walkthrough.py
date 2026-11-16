"""Compute a small offline report using native CPU execution."""

import numpy as np
import pandas as pd

from tabgen_eval import EvalData, Schema, evaluate

# prepare dataset
rng = np.random.default_rng(42)
holdout = pd.DataFrame(
    {
        "age": rng.normal(40, 10, 400),
        "income": rng.lognormal(10, 0.5, 400),
        "region": rng.choice(["north", "south", "west"], 400),
    }
)
generated = pd.DataFrame(
    {
        "age": rng.normal(43, 10, 200),
        "income": rng.lognormal(10.1, 0.5, 200),
        "region": rng.choice(["north", "south", "west"], 200),
    }
)
eval_data = EvalData(
    test=holdout,
    generated=generated,
    schema=Schema(numeric=["age", "income"], categorical=["region"]),
)

# get a report with captured column and pair scores.
report = evaluate(
    eval_data,
    metrics=["shape", "trend"],
    device="cpu",
    seed=42,
    uncertainty={"mode": "both", "n_replicates": 10},
    details=True,
)
print(report.summary())
