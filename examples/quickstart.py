"""Evaluate local synthetic examples against an independent real holdout.

Run with ``python examples/quickstart.py``; use ``--rows 40 --device cpu``
for a small offline smoke run. Replace the example frames with your own data.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tabgen_eval import EvalData, Schema, evaluate


def make_frame(n: int, *, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(40 + shift, 10, n).round(),
            "income": rng.gamma(2.0, 20_000.0, n),
            "job": rng.choice(["eng", "sales", "ops", "mgr"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "marital": rng.choice(["single", "married"], n),
        }
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000, help="real holdout rows")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    args = parser.parse_args(argv)
    if args.rows < 2:
        parser.error("rows must be at least two")
    holdout = make_frame(args.rows, seed=1)
    generated = make_frame(max(1, 3 * args.rows // 4), seed=2, shift=0.4)
    schema = Schema(
        numeric=["age", "income"],
        categorical=["job", "marital"],
    )
    report = evaluate(
        EvalData(test=holdout, generated=generated, schema=schema),
        device=args.device,
        seed=0,
        group=["fidelity", "conditional_fidelity", "geometric_fidelity"],
    )
    print(report.summary())


if __name__ == "__main__":
    main()
