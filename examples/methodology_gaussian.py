"""Regenerate the Gaussian methodology example and its exact finite-sample targets.

Run from the repository root with the project environment:

    PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        .venv/bin/python examples/methodology_gaussian.py

The paired example uses a fixed-seed real holdout from N(0, I_3), a generated
table of the form (Z, Z, W), and 10 repeated without-replacement comparisons.
Raw score percentiles describe comparison variability; they are not confidence
intervals for the expected score.
Analytic expectations are accompanied by one realized native metric score from
a fresh independent Gaussian sample pair at each comparison size.
"""

from __future__ import annotations

import argparse
import json
from math import comb, exp, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln

from tabgen_eval import EvalData, Schema, evaluate

COLUMNS = ("x", "y", "z")
METRICS = ("shape", "trend")
HIGHLIGHTS = (1000, 5000, 50000)
HOLDOUT_ROWS = 10000
GENERATED_ROWS = 10000
N_REPLICATES = 10
HOLDOUT_SEED = 42
SAMPLING_SEED = 43
REALIZATION_SEED = 44
GENERATED_SEED = 44
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs/assets/methodology-gaussian.json"


def shape_reference_mean(n: int) -> float:
    """Return 1 - E[D_n,n] from exact two-sided KS path counts.

    For continuous marginals, D_n,n takes values k/n. Its tail at k/n is
    2 / C(2n,n) times sum_j (-1)^(j-1) C(2n,n-jk). Summing these tails gives
    E[D_n,n]. Group equal products j*k and update binomial coefficients by
    recurrence so all counting arithmetic remains integral.
    """
    if n < 1:
        raise ValueError("Shape reference samples must contain at least one row")
    weights = [0] * (n + 1)
    for divisor in range(1, n + 1):
        sign = 1 if divisor % 2 else -1
        for multiple in range(divisor, n + 1, divisor):
            weights[multiple] += sign
    paths = comb(2 * n, n)
    coefficient = paths
    tail_sum = 0
    for offset in range(1, n + 1):
        coefficient = coefficient * (n - offset + 1) // (n + offset)
        tail_sum += weights[offset] * coefficient
    denominator = n * paths
    return (denominator - 2 * tail_sum) / denominator


def trend_reference_mean(n: int) -> float:
    """Return E[1 - |r-r'|/2] for independent Gaussian samples with rho=0.

    For n>2, U=(r+1)/2 is Beta(a,a), a=(n-2)/2, and
    E|U-V| = 2 B(2a,2a) / (a B(a,a)^2). For integer a, this equals
    C(2a,a)^2 / (2 C(4a,2a)), which can be evaluated with exact counts.
    """
    if n < 2:
        raise ValueError("Trend reference samples must contain at least two rows")
    if n == 2:
        return 0.5
    if n % 2 == 0:
        a = (n - 2) // 2
        denominator = 2 * comb(4 * a, 2 * a)
        numerator = comb(2 * a, a) ** 2
        return (denominator - numerator) / denominator
    a = (n - 2) / 2
    difference = exp(log(2) - log(a) + betaln(2 * a, 2 * a) - 2 * betaln(a, a))
    return 1 - difference


def target_sizes() -> list[int]:
    """Choose ten log-spaced sizes, including the three highlighted resolutions."""
    sizes = np.rint(np.geomspace(HIGHLIGHTS[0], HIGHLIGHTS[-1], 10)).astype(int)
    closest = int(np.argmin(np.abs(np.log(sizes) - np.log(HIGHLIGHTS[1]))))
    sizes[closest] = HIGHLIGHTS[1]
    return sorted(int(n) for n in sizes)


def realized_scores(n: int) -> dict[str, float]:
    """Compute native metric scores for one fresh independent Gaussian sample pair."""
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.metrics import Shape, Trend
    from tabgen_eval.preprocessing.columns import PreparedColumns

    rng = np.random.default_rng(np.random.SeedSequence([REALIZATION_SEED, n]))
    schema = Schema(numeric=list(COLUMNS))
    metrics = (Shape(device="cpu", dtype="float64"), Trend(device="cpu", dtype="float64"))
    frame = pd.DataFrame(rng.standard_normal((2 * n, len(COLUMNS))), columns=COLUMNS)
    first = np.arange(n)[None, :]
    second = first + n
    context = TorchContext.from_columns(
        PreparedColumns.prepare(frame, frame, schema), device="cpu", n_jobs=2
    )
    scores = {
        f"{name}_score": float(context.compute(metric, first, second, ideal=True)[name][0])
        for metric, name in zip(metrics, METRICS, strict=True)
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise RuntimeError(f"Expected finite realized metric scores at n={n}")
    return scores


def make_asset() -> dict:
    """Combine exact curves with paired real-real and generated-real estimates."""
    sizes = target_sizes()
    targets = [
        {
            "n": n,
            "shape": shape_reference_mean(n),
            "trend": trend_reference_mean(n),
            "highlight": n in HIGHLIGHTS,
        }
        for n in sizes
    ]
    holdout = pd.DataFrame(
        np.random.default_rng(HOLDOUT_SEED).standard_normal((HOLDOUT_ROWS, len(COLUMNS))),
        columns=COLUMNS,
    )
    latent = np.random.default_rng(GENERATED_SEED).standard_normal((GENERATED_ROWS, 2))
    generated = pd.DataFrame(
        {COLUMNS[0]: latent[:, 0], COLUMNS[1]: latent[:, 0], COLUMNS[2]: latent[:, 1]}
    )
    report = evaluate(
        EvalData(test=holdout, generated=generated, schema=Schema(numeric=list(COLUMNS))),
        metrics=METRICS,
        uncertainty={
            "mode": "both",
            "n_replicates": N_REPLICATES,
        },
        seed=SAMPLING_SEED,
        device="cpu",
        dtype="float64",
        n_jobs=2,
    )
    rows_per_sample = HOLDOUT_ROWS // 2
    scores, means, points, bands = {}, {}, {}, {}
    for role in ("reference", "generated"):
        scores[role], means[role], points[role], bands[role] = {}, {}, {}, {}
        for metric in METRICS:
            result = report[metric]
            assert result.n_eff == rows_per_sample, (
                f"Expected {rows_per_sample} rows per sample for {metric}, received {result.n_eff}"
            )
            distribution = getattr(result, f"{role}_distribution")
            if distribution is None:
                raise RuntimeError(f"Missing {role} repetitions for {metric}")
            samples = np.asarray(distribution.samples, dtype=np.float64)
            if samples.shape != (N_REPLICATES,) or not np.isfinite(samples).all():
                raise RuntimeError(f"Expected {N_REPLICATES} finite {role} scores for {metric}")
            raw_low, raw_high = np.quantile(samples, [0.025, 0.975], method="linear")
            np.testing.assert_allclose(
                [distribution.band_low, distribution.band_high],
                [raw_low, raw_high],
                rtol=0,
                atol=1e-15,
            )
            np.testing.assert_allclose(distribution.mean, samples.mean(), rtol=0, atol=1e-15)
            point = result.ideal if role == "reference" else result.value
            low, high = (
                (result.ideal_ci_low, result.ideal_ci_high)
                if role == "reference"
                else (result.ci_low, result.ci_high)
            )
            np.testing.assert_allclose([low, high], [raw_low, raw_high], rtol=0, atol=1e-15)
            if not np.isfinite([point, low, high]).all():
                raise RuntimeError(f"Expected a finite point and interval for {role} {metric}")
            scores[role][metric] = samples.tolist()
            means[role][metric] = float(distribution.mean)
            points[role][metric] = float(point)
            bands[role][metric] = {"low": float(low), "high": float(high)}
    exact = next(target for target in targets if target["n"] == rows_per_sample)
    for index, target in enumerate(targets, start=1):
        target.update(realized_scores(target["n"]))
        if index == 1 or index % 5 == 0 or index == len(targets):
            print(
                f"Realized scores: {index}/{len(targets)} sizes (n={target['n']:,})",
                flush=True,
            )
    return {
        "population": {
            "distribution": "multivariate_normal",
            "mean": [0, 0, 0],
            "covariance": np.eye(len(COLUMNS), dtype=int).tolist(),
            "columns": list(COLUMNS),
        },
        "realizations": {
            "pairs_per_size": 1,
            "seed": REALIZATION_SEED,
            "sampling": "one fresh independent Gaussian sample pair at each resolution",
        },
        "targets": targets,
        "generator": {
            "distribution": "singular_multivariate_normal",
            "construction": "(Z, Z, W), with independent standard-normal Z and W",
            "mean": [0, 0, 0],
            "covariance": [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
            "columns": list(COLUMNS),
            "seed": GENERATED_SEED,
        },
        "resampling": {
            "holdout_rows": HOLDOUT_ROWS,
            "generated_rows": GENERATED_ROWS,
            "rows_per_sample": rows_per_sample,
            "n_replicates": N_REPLICATES,
            "band_method": "score_percentiles",
            "sampling": "without replacement within each comparison",
            "holdout_seed": HOLDOUT_SEED,
            "generated_seed": GENERATED_SEED,
            "sampling_seed": SAMPLING_SEED,
            "device": "cpu",
            "dtype": "float64",
            "n_jobs": 2,
            "scores": scores,
            "mean": means,
            "point": points,
            "band": bands,
            "exact_mean": {
                "reference": {metric: exact[metric] for metric in METRICS},
                "generated": {
                    "shape": exact["shape"],
                    "trend": (0.5 + 2 * exact["trend"]) / 3,
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asset = make_asset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asset, indent=2, allow_nan=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
