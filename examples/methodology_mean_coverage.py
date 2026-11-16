"""Summarize reference-mean coverage from saved CPU synthetic studies.

Calibration folders contain scores.npz with a calibration array (holdouts,
replicates, metrics). Independent population scores come from fresh_real_real
in those files, or fresh_population_scores in a separate study folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("shape", "trend")


def summarize(calibration, population_scores, *, population, n, budgets):
    """Retain every interval and count coverage across independent holdouts."""
    if calibration.ndim != 3 or calibration.shape[2] != len(METRICS):
        raise ValueError("Expected calibration scores with shape (holdouts, B, 2)")
    population_scores = population_scores.reshape(-1, len(METRICS))
    if not np.isfinite(calibration).all() or not np.isfinite(population_scores).all():
        raise ValueError("This synthetic study requires finite scores")
    target = population_scores.mean(axis=0)
    target_se = population_scores.std(axis=0, ddof=1) / np.sqrt(len(population_scores))
    if len(population_scores) < 16:
        raise ValueError("At least 16 independent population pairs are needed for the illustration")
    batches = np.array_split(population_scores, 16)
    mean_estimates = np.stack([batch.mean(axis=0) for batch in batches])
    rows = []
    for budget in budgets:
        if not 2 <= budget <= calibration.shape[1]:
            raise ValueError(f"Replicate budget {budget} exceeds available scores or is below 2")
        scores = calibration[:, :budget]
        low, high = np.quantile(scores, [0.025, 0.975], axis=1, method="linear")
        points = scores.mean(axis=1)
        for index, metric in enumerate(METRICS):
            inside = (low[:, index] <= target[index]) & (target[index] <= high[:, index])
            count, rate = len(inside), float(inside.mean())
            z = 1.959963984540054
            denominator = 1 + z**2 / count
            center = (rate + z**2 / (2 * count)) / denominator
            radius = z * np.sqrt(rate * (1 - rate) / count + z**2 / (4 * count**2)) / denominator
            rows.append(
                {
                    "population": population,
                    "n_res": n,
                    "N": 2 * n,
                    "metric": metric,
                    "B": budget,
                    "holdouts": count,
                    "mu_estimate": float(target[index]),
                    "mu_estimate_mc_se": float(target_se[index]),
                    "population_score_pairs": len(population_scores),
                    "independent_mean_estimates": mean_estimates[:, index].tolist(),
                    "pairs_per_mean_estimate": [len(batch) for batch in batches],
                    "covered": int(inside.sum()),
                    "coverage": rate,
                    "coverage_wilson95": [center - radius, center + radius],
                    "mean_width": float((high - low)[:, index].mean()),
                    "intervals": {
                        "mean": points[:, index].tolist(),
                        "low": low[:, index].tolist(),
                        "high": high[:, index].tolist(),
                    },
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--population-scores", type=Path)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1000, 5000, 50000])
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 500])
    parser.add_argument("--output", type=Path, default=Path("methodology-mean-coverage.json"))
    args = parser.parse_args()
    rows = []
    for n in args.resolutions:
        for population in ("gaussian", "balanced_mixed"):
            case = f"{population}-n{n}"
            with np.load(args.calibration / case / "scores.npz") as data:
                calibration = data["calibration"].copy()
                if args.population_scores is None:
                    population_scores = data["fresh_real_real"].copy()
            if args.population_scores is not None:
                with np.load(args.population_scores / case / "scores.npz") as data:
                    population_scores = data["fresh_population_scores"].copy()
            rows.extend(
                summarize(
                    calibration, population_scores, population=population, n=n, budgets=args.budgets
                )
            )
    result = {
        "design": {
            "target": "mu_n(P) = E[T(R'_n, R_n)] for independent population samples",
            "nominal_coverage": 0.95,
            "sampling": "Production calibration: fresh disjoint samples without replacement from each holdout",
            "interval": "Raw linear 2.5th and 97.5th percentiles; no recentering",
            "comparison": "Common replicate prefixes on identical holdouts across budgets",
            "population_target": "Estimated using independent population pairs; MC SE retained",
            "coverage_unit": "One independently generated calibration holdout",
            "scope": "Two regular synthetic populations, Shape and Trend; empirical evidence, not a general coverage guarantee",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
