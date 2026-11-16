"""Evaluate generated data against an independent OpenML holdout.

Run ``python examples/openml_quickstart.py [dataset]`` (default: adult, id 179).
The first run downloads the dataset; subsequent runs can use its local cache.
The noised training resample illustrates the API and is not a trained generator.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tabgen_eval import EvalData, Schema, evaluate, load_openml


def toy_synthetic(train: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Resample training rows and jitter numeric columns for the tutorial."""
    rng = np.random.default_rng(seed)
    generated = train.sample(n=len(train), replace=True, random_state=seed).reset_index(drop=True)
    for col in generated.columns:
        values = generated[col]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            std = 1.0
        noised = values + rng.normal(0.0, 0.05 * std, len(values))
        if pd.api.types.is_integer_dtype(values):
            noised = noised.round().astype(values.dtype)
        generated[col] = noised
    return generated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="179")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    args = parser.parse_args(argv)
    dataset = int(args.dataset) if args.dataset.isdecimal() else args.dataset
    train, holdout = load_openml(dataset)
    schema = Schema.infer(train)
    # Check these inferred roles against the dataset's meaning before using real data.
    generated = toy_synthetic(train)
    print(
        f"OpenML {dataset}: train={len(train)}, holdout={len(holdout)}, columns={holdout.shape[1]}"
    )
    print(
        "Tutorial generated pool: noised training resample; "
        f"n_res={min(len(generated), len(holdout) // 2)}"
    )
    report = evaluate(
        EvalData(test=holdout, generated=generated, schema=schema),
        device=args.device,
        seed=0,
        uncertainty={"mode": "reference", "n_replicates": args.n_replicates},
        group=["fidelity", "conditional_fidelity", "geometric_fidelity"],
    )
    print(report.summary())


if __name__ == "__main__":
    main()
