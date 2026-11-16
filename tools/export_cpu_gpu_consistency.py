"""Export saved Torch CPU/CUDA scores and a five-dataset consistency table.

No metrics are rerun. The export cross-checks native_parity.csv against the
original scores.csv and retains nonfinite statuses rather than treating them
as zero numerical differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("video_transcoding_time_prediction", "Video"),
    ("hotel_booking_demand", "Hotel"),
    ("allstate_claims_severity", "Allstate"),
    ("covertype", "Covertype"),
    ("mercari_price_suggestion", "Mercari"),
)
SAMPLES = ("test_split", "ours_15h", "tabdiff_15h", "ltm_step378000")
ROWS = (
    ("shape", None, "Shape"),
    ("trend", None, "Trend"),
    ("precision", None, "Precision"),
    ("recall", None, "Recall"),
    ("density", None, "Density"),
    ("coverage", None, "Coverage"),
    ("cond_shape", None, "Conditional Shape"),
    ("cond_trend", None, "Conditional Trend"),
    ("recovery", None, "Recovery"),
    ("concentration", None, "Concentration"),
    ("linear_c2st", "auc", "Linear C2ST AUC"),
    ("nonlinear_c2st", "auc", "Nonlinear C2ST AUC"),
    ("dcr_ratio", "ratio", "DCR ratio"),
    ("near_match_rate", "rate", "Near-match Rate"),
    ("replication_ratio", "rate", "Replication Ratio"),
    ("mle", "score", "MLE score (AUROC / MAE)"),
    ("mle", "ratio", "MLE ratio"),
    ("mle", "prediction_agreement", "MLE prediction agreement"),
    ("mle", "prediction_mse", "MLE prediction MSE"),
)
KEYS = ("dataset", "sample", "computation", "metric", "stat")
ALLOWED = {"PASS", "ESTIMATOR_DEVICE_VARIATION", "BOTH_ABSENT", "BOTH_UNDEFINED"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "profiling_20261022")
    args = parser.parse_args()
    source = args.source
    manifest = json.loads((source / "manifest.json").read_text())
    parity = read_csv(source / "native_parity.csv")
    scores = read_csv(source / "scores.csv")
    lookup = {}
    for row in scores:
        key = tuple(row[k] for k in KEYS) + (row["configuration"],)
        if key in lookup:
            raise ValueError(f"Duplicate source score: {key}")
        lookup[key] = row
    pairs, seen = [], set()
    for row in parity:
        identity = tuple(row[k] for k in KEYS)
        key = identity + (row["field"],)
        if key in seen:
            raise ValueError(f"Duplicate CPU/GPU comparison: {key}")
        seen.add(key)
        assert row["configuration"] == "cuda" and row["reference_configuration"] == "torch_cpu"
        assert row["comparison_status"] in ALLOWED
        assert row["metadata_equal"] == "True" and row["not_converged"] == "False"
        pair = {k: row[k] for k in KEYS}
        pair.update(
            field=row["field"],
            comparison_status=row["comparison_status"],
            numerical_status=row["numerical_status"],
            cpu_status=row["reference_value_status"],
            gpu_status=row["actual_value_status"],
            cpu_flags=row["reference_flags"],
            gpu_flags=row["actual_flags"],
            n_eff=int(float(row["n_eff"])) if row["n_eff"] else None,
            cpu_value=None,
            gpu_value=None,
            absolute_difference=None,
            relative_difference_cpu=None,
        )
        for configuration, prefix, recorded in (
            ("torch_cpu", "cpu", "reference"),
            ("cuda", "gpu", "actual"),
        ):
            original = lookup[identity + (configuration,)]
            assert original[row["field"] + "_status"] == pair[prefix + "_status"]
            if pair[prefix + "_status"] == "finite":
                value = float(original[row["field"]])
                assert math.isfinite(value) and value == float(row[recorded + "_value"])
                pair[prefix + "_value"] = value
        if pair["cpu_value"] is not None and pair["gpu_value"] is not None:
            delta = abs(pair["gpu_value"] - pair["cpu_value"])
            assert math.isclose(delta, float(row["abs_error"]), rel_tol=1e-12, abs_tol=1e-18)
            pair["absolute_difference"] = delta
            pair["relative_difference_cpu"] = (
                delta / abs(pair["cpu_value"])
                if pair["cpu_value"]
                else (0.0 if delta == 0 else None)
            )
        pairs.append(pair)
    expected = {
        tuple(row[k] for k in KEYS) + (field,)
        for row in scores
        if row["configuration"] == "torch_cpu"
        for field in ("value", "ideal")
    }
    assert seen == expected and len(pairs) == 1690
    assert {r["dataset"] for r in pairs} == {d for d, _ in DATASETS}
    assert {r["sample"] for r in pairs} == set(SAMPLES)
    assert len({r["metric"] for r in pairs}) == 16

    matrix = []
    for metric, stat, label in ROWS:
        cells = {}
        for dataset, _ in DATASETS:
            selected = [
                r
                for r in pairs
                if r["dataset"] == dataset
                and r["metric"] == metric
                and (stat is None or r["stat"] == stat)
            ]
            assert selected, (metric, stat, dataset)
            finite = [r for r in selected if r["absolute_difference"] is not None]
            if finite:
                worst = max(finite, key=lambda r: r["absolute_difference"])
                cell = {
                    "status": "finite",
                    "max_absolute_difference": max(r["absolute_difference"] for r in finite),
                    "finite_pairs": len(finite),
                    "worst_pair": {
                        key: worst[key]
                        for key in (
                            "sample",
                            "stat",
                            "field",
                            "cpu_value",
                            "gpu_value",
                            "absolute_difference",
                            "relative_difference_cpu",
                        )
                    },
                }
            elif any("ZERO_BASELINE" in r["cpu_flags"] for r in selected):
                cell = {
                    "status": "zero_baseline",
                    "max_absolute_difference": None,
                    "finite_pairs": 0,
                }
            elif any("NOT_APPLICABLE" in r["cpu_flags"] for r in selected):
                cell = {
                    "status": "not_applicable",
                    "max_absolute_difference": None,
                    "finite_pairs": 0,
                }
            else:
                raise ValueError(f"Unexpected lack of finite results: {metric}/{stat}/{dataset}")
            cell["matching_absent_pairs"] = sum(
                r["comparison_status"] == "BOTH_ABSENT" for r in selected
            )
            cell["matching_undefined_pairs"] = sum(
                r["comparison_status"] == "BOTH_UNDEFINED" for r in selected
            )
            cells[dataset] = cell
        matrix.append({"metric": metric, "stat": stat, "label": label, "cells": cells})

    deterministic = [
        r
        for r in pairs
        if r["metric"] not in {"linear_c2st", "nonlinear_c2st", "mle"}
        and r["absolute_difference"] is not None
    ]
    summary = {
        "metrics": 16,
        "datasets": 5,
        "sample_sources": 4,
        "scalar_pairs": len(pairs),
        "finite_pairs": sum(r["absolute_difference"] is not None for r in pairs),
        "comparison_status_counts": dict(Counter(r["comparison_status"] for r in pairs)),
        "deterministic_metric_count": len({r["metric"] for r in deterministic}),
        "deterministic_max_absolute_difference": max(
            r["absolute_difference"] for r in deterministic
        ),
        "linear_c2st_max_absolute_difference": max(
            r["absolute_difference"]
            for r in pairs
            if r["metric"] == "linear_c2st" and r["absolute_difference"] is not None
        ),
        "nonlinear_c2st_max_absolute_difference": max(
            r["absolute_difference"]
            for r in pairs
            if r["metric"] == "nonlinear_c2st" and r["absolute_difference"] is not None
        ),
    }
    asset = {
        "context": {
            "recorded_date": "2026-10-22",
            "source_revision": manifest["commit"],
            "cpu_configuration": "torch_cpu",
            "gpu_configuration": "cuda",
            "dtype": "float64",
            "seed": 42,
            "n_jobs": 8,
            "datasets": [{"dataset": d, "label": label} for d, label in DATASETS],
            "samples": list(SAMPLES),
            "source_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("scores.csv", "native_parity.csv")
            },
            "aggregation": "Maximum absolute GPU-minus-CPU difference across four sample sources, point value and real-real ideal, and all component statistics in the indicated row. DCR ratio and individual MLE statistics are separate rows. No averaging or pooling across datasets or incompatible target units.",
            "interpretation": "Deterministic numerical parity, linear-solver agreement, and XGBoost fitted-model device variation are distinct. Matching undefined or absent results are not finite agreement. Regression MAE and prediction MSE retain each dataset's target units and squared units, respectively.",
        },
        "summary": summary,
        "matrix": matrix,
        "records": pairs,
    }
    destination = ROOT / "docs/assets"
    header = json.dumps({k: v for k, v in asset.items() if k != "records"}, indent=2)
    (destination / "cpu-gpu-consistency.json").write_text(
        header[:-2]
        + ',\n  "records": [\n'
        + ",\n".join("    " + json.dumps(r, separators=(",", ":"), allow_nan=False) for r in pairs)
        + "\n  ]\n}\n"
    )
    write_csv(destination / "cpu-gpu-values.csv", pairs, list(pairs[0]))
    matrix_rows = [
        {
            "metric": row["label"],
            **{
                label: row["cells"][dataset]["max_absolute_difference"]
                if row["cells"][dataset]["status"] == "finite"
                else row["cells"][dataset]["status"]
                for dataset, label in DATASETS
            },
        }
        for row in matrix
    ]
    write_csv(destination / "cpu-gpu-consistency.csv", matrix_rows, list(matrix_rows[0]))
    worst_rows = [
        {
            "metric": row["label"],
            "dataset": dataset,
            "status": row["cells"][dataset]["status"],
            **row["cells"][dataset].get("worst_pair", {}),
        }
        for row in matrix
        for dataset, _ in DATASETS
    ]
    write_csv(
        destination / "cpu-gpu-largest-differences.csv",
        worst_rows,
        [
            "metric",
            "dataset",
            "status",
            "sample",
            "stat",
            "field",
            "cpu_value",
            "gpu_value",
            "absolute_difference",
            "relative_difference_cpu",
        ],
    )
    lines = [
        "# CPU/GPU score consistency",
        "",
        "Recorded October 22, 2026 at revision `"
        + manifest["commit"][:12]
        + "`; optimized Torch CPU versus CUDA, FP64, seed 42, n_jobs=8.",
        "",
        "Cells show the largest absolute difference across four sample sources and both point/reference fields, including component statistics. All 16 metrics are covered; MLE outputs are separated by units. Raw CPU and GPU values are in [cpu-gpu-values.csv](../docs/assets/cpu-gpu-values.csv).",
        "",
        "| Metric / statistic | " + " | ".join(label for _, label in DATASETS) + " |",
        "| --- | " + " | ".join("---:" for _ in DATASETS) + " |",
    ]
    for row in matrix:
        values = []
        for dataset, _ in DATASETS:
            cell = row["cells"][dataset]
            values.append(
                f"{cell['max_absolute_difference']:.3g}"
                if cell["status"] == "finite"
                else ("Undefined (zero baseline)" if cell["status"] == "zero_baseline" else "N/A")
            )
        lines.append("| " + row["label"] + " | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Video = Video Transcoding Time Prediction; Hotel = Hotel Booking Demand; Allstate = Allstate Claims Severity; Mercari = Mercari Price Suggestion.",
            "",
            "Deterministic metrics agree to floating-point precision. Linear C2ST has small solver differences. Nonlinear C2ST and MLE train separate XGBoost models on CPU and GPU; their differences are fitted-model variation, not a kernel-parity guarantee. MLE score uses AUROC on classification and MAE on regression; prediction MSE has squared target units.",
            "",
            "Mercari DCR ratio is undefined on both devices because the real holdout median distance is zero. Its finite distance quantiles agree exactly; this is not a finite ratio match. N/A denotes prediction agreement on regression or prediction MSE on classification.",
            "",
        ]
    )
    (source / "cpu_gpu_consistency.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    print("\n".join(lines[6:28]))


if __name__ == "__main__":
    main()
