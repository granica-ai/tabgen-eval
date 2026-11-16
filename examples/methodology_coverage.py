"""Measure real-reference band coverage using independent population comparisons.

Each outer trial constructs a new holdout, calibrates the actual package metric,
and tests its band on fresh real-real scores. The default experiment is small;
use the documented options to extend the sample sizes or replication budget.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabgen_eval import EvalData, Schema, evaluate

POPULATIONS = ("gaussian", "balanced_mixed")
BUDGETS = (10, 50, 200, 500)
METRICS = ("shape", "trend")


def population(rng, size, case):
    """Draw from a regular correlated Gaussian or a balanced mixed population."""
    u, v, w = rng.normal(size=(3, size))
    x, y, z = u, 0.5 * u + np.sqrt(0.75) * v, 0.3 * u + np.sqrt(0.91) * w
    if case == "gaussian":
        return pd.DataFrame({"x": x, "y": y, "z": z}), Schema(numeric=["x", "y", "z"])
    c = rng.integers(0, 4, size=size)
    return pd.DataFrame({"x": x + 0.3 * (c - 1.5), "y": y + 0.3 * (c - 1.5), "c": c}), Schema(
        numeric=["x", "y"], categorical=["c"]
    )


def independent_scores(rng, n, case, count, threads):
    """Compute independent fresh pairs with the same compiled metric operators."""
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.metrics import Shape, Trend
    from tabgen_eval.preprocessing.columns import PreparedColumns

    metrics = (Shape(device="cpu"), Trend(device="cpu"))
    arrays = []
    for start in range(0, count, 8):
        batch = min(8, count - start)
        frame, schema = population(rng, 2 * n * batch, case)
        offsets = 2 * n * np.arange(batch)[:, None]
        first = offsets + np.arange(n)[None, :]
        second = first + n
        context = TorchContext.from_columns(
            PreparedColumns.prepare(frame, frame, schema), device="cpu", n_jobs=threads
        )
        arrays.append(
            np.column_stack(
                [
                    context.compute(metric, first, second, ideal=True)[name]
                    for metric, name in zip(metrics, METRICS, strict=True)
                ]
            )
        )
    return np.vstack(arrays)


def write_illustration(output, case, n, calibration, fresh):
    """Save the first holdout's bootstrap scores and independent population draws."""
    illustration = {
        "population": case,
        "N": 2 * n,
        "n_res": n,
        "calibration_trial": 0,
        "selection": "First holdout and all fresh independent pairs; no trial selection by agreement",
        "samples": {
            name: {
                "bootstrap": calibration[0, :, index].tolist(),
                "fresh_real_real": fresh[:, :, index].ravel().tolist(),
            }
            for index, name in enumerate(METRICS)
        },
    }
    (output / "illustration.json").write_text(json.dumps(illustration, indent=2) + "\n")


def run_group(args, case, n, outer):
    """Keep outer holdouts independent and use common replicate prefixes across B."""
    output = args.output / f"{case}-n{n}"
    output.mkdir(parents=True, exist_ok=True)
    raw = np.empty((outer, max(args.budgets), len(METRICS)))
    oracle = np.empty((outer, args.fresh_pairs, len(METRICS)))
    started = time.perf_counter()
    for trial in range(outer):
        key = [args.seed, POPULATIONS.index(case), n, trial]
        real, schema = population(
            np.random.default_rng(np.random.SeedSequence(key + [0])), 2 * n, case
        )
        reference = evaluate(
            EvalData(test=real, schema=schema),
            metrics=METRICS,
            device="cpu",
            n_jobs=args.threads,
            dtype="float64",
            seed=key + [1],
            uncertainty={"mode": "reference", "n_replicates": max(args.budgets)},
        )
        for index, name in enumerate(METRICS):
            assert reference.provenance["metric_execution"][name]["implementation"] == "aten_cpp"
            raw[trial, :, index] = reference[name].reference_distribution.samples
        oracle[trial] = independent_scores(
            np.random.default_rng(np.random.SeedSequence(key + [2])),
            n,
            case,
            args.fresh_pairs,
            args.threads,
        )
        if trial == 0 or (trial + 1) % 16 == 0 or trial + 1 == outer:
            print(
                f"{case} n={n:,}: {trial + 1}/{outer} independent holdouts, {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    assert np.isfinite(raw).all() and np.isfinite(oracle).all()
    np.savez_compressed(output / "scores.npz", calibration=raw, fresh_real_real=oracle)
    write_illustration(output, case, n, raw, oracle)
    rows = []
    for budget in args.budgets:
        limits = np.quantile(raw[:, :budget, :], [0.025, 0.975], axis=1, method="linear")
        contained = (oracle >= limits[0][:, None, :]) & (oracle <= limits[1][:, None, :])
        per_holdout = contained.mean(axis=1)
        for index, name in enumerate(METRICS):
            coverage = float(per_holdout[:, index].mean())
            se = float(per_holdout[:, index].std(ddof=1) / np.sqrt(outer))
            values = oracle[:, :, index].ravel()
            rows.append(
                dict(
                    population=case,
                    metric=name,
                    N=2 * n,
                    n_res=n,
                    B=budget,
                    outer_holdouts=outer,
                    fresh_pairs_per_holdout=args.fresh_pairs,
                    coverage=coverage,
                    monte_carlo_se=se,
                    mc_low=max(0.0, coverage - 1.96 * se),
                    mc_high=min(1.0, coverage + 1.96 * se),
                    fresh_score_mean=float(values.mean()),
                    fresh_score_mean_se=float(values.std(ddof=1) / np.sqrt(values.size)),
                    fresh_score_q025=float(np.quantile(values, 0.025)),
                    fresh_score_q975=float(np.quantile(values, 0.975)),
                )
            )
    result = dict(population=case, n_res=n, seconds=time.perf_counter() - started, rows=rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--outer", type=int, default=128)
    parser.add_argument("--large-outer", type=int, default=48)
    parser.add_argument("--fresh-pairs", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path, default=Path("coverage-results"))
    args = parser.parse_args()
    if (
        min(args.resolutions) < 2
        or min(args.budgets) < 2
        or min(args.outer, args.large_outer) < 2
        or args.fresh_pairs < 1
        or args.threads < 1
    ):
        parser.error("resolutions, budgets and outer counts must be >=2; pairs and threads >=1")
    import torch

    def forbidden(*unused, **kwargs):
        raise RuntimeError("GPU probing is disabled during this CPU-only study")

    for name in ("is_available", "device_count", "_lazy_init"):
        setattr(torch.cuda, name, forbidden)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "design": {
            "target": "Coverage of a fresh independent real-real score by the raw central95% reference band",
            "nominal_coverage": 0.95,
            "real_pool_size": "N=2*n_res",
            "sampling": "Production test-only evaluate: disjoint real draws without replacement, full source eligible in each replicate",
            "outer_holdouts": args.outer,
            "large_outer_holdouts": args.large_outer,
            "fresh_pairs_per_holdout": args.fresh_pairs,
            "budgets": args.budgets,
            "threads": args.threads,
            "precision": "float64",
            "implementation": "ATen/C++ CPU",
            "seed": args.seed,
            "mc_error": "Normal Monte Carlo intervals from independent outer-holdout coverage means; dependent within-holdout checks are not treated as independent trials",
        },
        "groups": [],
    }
    for n in args.resolutions:
        for case in POPULATIONS:
            outer = args.large_outer if n >= 50000 else args.outer
            result["groups"].append(run_group(args, case, n, outer))
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Results:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
