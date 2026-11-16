"""Compare compiled CPU and CUDA through the public evaluation API.

Each case uses the same sources, schema, automatic resolution, seed and registered metric defaults.
Fixed evaluation computes generated-real and disjoint real-real scores. Timings
include validation, preparation, transfers and numerical work; imports, file loading
and device discovery precede timing. First-call timings retain existing caches.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from tabgen_eval import EvalData, Schema, __version__, evaluate
from tabgen_eval.schema import infer_schema

METRICS = ("shape", "trend", "conditional_fidelity", "linear_c2st")


def make(n, numeric, categorical, cardinality, seed):
    """Build numeric and categorical columns from one private random stream."""
    rng = np.random.default_rng(seed)
    nc, cc = [f"n{i}" for i in range(numeric)], [f"c{i}" for i in range(categorical)]
    table = pd.DataFrame(rng.normal(size=(n, numeric)), columns=nc)
    for column in cc:
        table[column] = rng.integers(cardinality, size=n).astype(str)
    return table, Schema(numeric=nc, categorical=cc)


def no_sync():
    """CPU calls return after their host-visible work completes."""


def json_safe(value):
    """Represent noncomputable floating scores as JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def measure(fn, *, synchronize=no_sync, repeats, torch=None, device=None):
    """Time one first invocation and subsequent warm calls, synchronizing CUDA.

    Import, data loading and device discovery precede this helper. Existing
    process/library/JIT caches are retained; the first call is not process-cold.
    CUDA peak-extra allocation covers warm calls and excludes their initial
    resident storage and allocations outside PyTorch's allocator.
    """
    synchronize()
    start = perf_counter()
    value = fn()
    synchronize()
    first = perf_counter() - start
    baseline = peak = None
    if torch is not None and device is not None and device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        synchronize()
        start = perf_counter()
        value = fn()
        synchronize()
        times.append(perf_counter() - start)
    if baseline is not None:
        peak = torch.cuda.max_memory_allocated(device) - baseline
    return value, {
        "first_call_s": first,
        "median_s": float(np.median(times)),
        "runs_s": times,
        "peak_extra_bytes": peak,
    }


def cpu_information():
    """Record the CPU model and currently available logical CPUs when discoverable."""
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def library_versions():
    """Inspect installed distribution metadata without importing optional engines."""
    versions = {}
    for name in ("tabgen-eval", "numpy", "pandas", "scipy", "scikit-learn", "torch", "triton"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = __version__ if name == "tabgen-eval" else None
    return versions


def summarize(report, metric):
    """Keep scores, actual execution and classifier convergence visible together."""
    result = report.results[0]
    stats = {name: stat for item in report.results for name, stat in item.stats.items()}
    provenance = json.loads(report.to_json())["provenance"]
    return {
        "scores": {name: stat.value for name, stat in stats.items()},
        "real_real_scores": {name: stat.ideal for name, stat in stats.items()},
        "flags": {name: stat.flag_codes for name, stat in stats.items()},
        "n_res": next(iter(result.stats.values())).n_eff,
        "execution": provenance["procedures"]["comparison"]["metric_execution"][metric],
        "classifier_diagnostics": provenance["procedures"]["comparison"]
        .get("classifier_diagnostics", {})
        .get(metric),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "cuda"],
        default=["cpu", "cuda"],
    )
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--device", default="cuda:0", help="CUDA device for the cuda case")
    parser.add_argument(
        "--jobs", type=int, default=1, help="CPU numerical thread budget for every case"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="native numeric precision",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="synthetic rows in each source; resolution defaults to rows/2",
    )
    parser.add_argument("--columns", type=int, default=6, help="synthetic feature count")
    parser.add_argument(
        "--categorical", type=int, default=2, help="synthetic categorical feature count"
    )
    parser.add_argument("--cardinality", type=int, default=5)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--schema", type=Path, help="optional Schema JSON for supplied files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeat", type=int, default=3, help="warm invocations after the first call"
    )
    parser.add_argument("--uncertainty", choices=["none", "reference", "both"], default="none")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--out", type=Path, help="optional JSON artifact")
    args = parser.parse_args(argv)
    if (
        min(
            args.rows,
            args.columns,
            args.cardinality,
            args.jobs,
            args.repeat,
            args.n_replicates,
        )
        < 1
        or args.seed < 0
    ):
        parser.error(
            "rows/columns/cardinality/jobs/repeat/n-replicates must be positive; seed nonnegative"
        )
    if not 0 <= args.categorical <= args.columns:
        parser.error("categorical must be between zero and columns")
    if (args.holdout is None) != (args.generated is None):
        parser.error("supply both --holdout and --generated")
    if args.schema is not None and args.holdout is None:
        parser.error("--schema requires supplied files")
    declared = Schema.from_dict(json.loads(args.schema.read_text())) if args.schema else None
    if args.holdout:
        holdout, generated = (
            pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
            for path in (args.holdout, args.generated)
        )
        schema = declared or infer_schema(holdout)
    else:
        holdout, schema = make(
            args.rows,
            args.columns - args.categorical,
            args.categorical,
            args.cardinality,
            args.seed,
        )
        generated, _ = make(
            args.rows,
            args.columns - args.categorical,
            args.categorical,
            args.cardinality,
            args.seed + 1,
        )
    holdout_half = len(holdout) // 2
    if args.uncertainty == "both":
        n_res = holdout_half if len(generated) > holdout_half else len(generated) // 2
    else:
        n_res = min(holdout_half, len(generated))
    if len(holdout) < 2 or len(generated) < 1:
        parser.error("holdout must have at least two rows; generated must be nonempty")
    if "linear_c2st" in args.metrics and n_res < 3:
        parser.error("Linear C2ST needs at least three rows per class at the requested resolution")
    torch = None
    if args.devices:
        import torch
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_information(),
    }
    gpu = None
    cuda_reason = None
    if "cuda" in args.devices:
        device = torch.device(args.device)
        if device.type != "cuda":
            parser.error("--device must select cuda or cuda:<index>")
        if not torch.cuda.is_available():
            cuda_reason = "CUDA is unavailable in this environment"
        else:
            properties = torch.cuda.get_device_properties(device)
            gpu = {
                "device": str(device),
                "model": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(device)),
                "torch_cuda_version": torch.version.cuda,
            }
    hardware["gpu"] = gpu
    artifact = {
        "scope": "public_evaluate_one_metric",
        "timing": "first invocation of each case and subsequent warm invocations in one process; existing caches retained; imports/loading/device discovery excluded",
        "hardware": hardware,
        "libraries": library_versions(),
        "inputs": {
            "holdout": str(args.holdout) if args.holdout else "synthetic",
            "generated": str(args.generated) if args.generated else "synthetic",
            "n_holdout": len(holdout),
            "n_generated": len(generated),
            "schema": schema.to_dict(),
            "n_res": n_res,
        },
        "settings": {
            "seed": args.seed,
            "jobs": args.jobs,
            "native_dtype": args.dtype,
            "uncertainty": None
            if args.uncertainty == "none"
            else {"mode": args.uncertainty, "n_replicates": args.n_replicates},
        },
        "cases": [],
    }
    previous_threads = torch.get_num_threads() if torch is not None else None
    try:
        if torch is not None:
            torch.set_num_threads(args.jobs)
        with threadpool_limits(args.jobs):
            for metric in dict.fromkeys(args.metrics):
                for backend in dict.fromkeys(args.devices):
                    record = {"metric": metric, "case": backend}
                    if backend == "cuda" and cuda_reason:
                        artifact["cases"].append(
                            {**record, "status": "unavailable", "reason": cuda_reason}
                        )
                        continue
                    options = dict(
                        device=args.device if backend == "cuda" else "cpu",
                        dtype=args.dtype,
                        n_jobs=args.jobs,
                        seed=args.seed,
                        uncertainty=None
                        if args.uncertainty == "none"
                        else {"mode": args.uncertainty, "n_replicates": args.n_replicates},
                    )
                    synchronize = (
                        partial(torch.cuda.synchronize, args.device)
                        if backend == "cuda"
                        else no_sync
                    )
                    first = None

                    def evaluate_once(metric=metric, options=options):
                        nonlocal first
                        report = evaluate(
                            EvalData(test=holdout, generated=generated, schema=schema),
                            metrics=["cond_shape", "cond_trend", "recovery"]
                            if metric == "conditional_fidelity"
                            else [metric],
                            **options,
                        )
                        if first is None:
                            first = report
                        return report

                    report, timing = measure(
                        evaluate_once,
                        synchronize=synchronize,
                        repeats=args.repeat,
                        torch=torch,
                        device=torch.device(args.device) if backend == "cuda" else None,
                    )
                    record.update(
                        status="ok",
                        **summarize(report, metric),
                        first_call_scores={
                            name: stat.value
                            for item in first.results
                            for name, stat in item.stats.items()
                        },
                        actual_numeric_dtype=args.dtype,
                        first_call_s=timing["first_call_s"],
                        warm_median_s=timing["median_s"],
                        warm_runs_s=timing["runs_s"],
                        warm_peak_extra_torch_bytes=timing["peak_extra_bytes"],
                    )
                    artifact["cases"].append(json_safe(record))
                    if args.out:
                        args.out.write_text(
                            json.dumps(json_safe(artifact), indent=2, allow_nan=False)
                        )
    finally:
        if torch is not None:
            torch.set_num_threads(previous_threads)
    output = json.dumps(json_safe(artifact), indent=2, allow_nan=False)
    print(output)
    if args.out:
        args.out.write_text(output)


if __name__ == "__main__":
    main()
