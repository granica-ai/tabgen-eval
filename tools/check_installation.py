"""Exercise an installed wheel, including its compiler-free tensor binaries.

Run from outside the checkout with PYTHONPATH unset. This is deliberately a full
report check: preparation, dispatch, point/real-reference estimates, refits, replicate
arrays, provenance and JSON serialization all use the installed package.
"""

from __future__ import annotations

import argparse
import importlib.abc
import json
import os
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--without-compiler", action="store_true")
    parser.add_argument("--block-torch", action="store_true")
    parser.add_argument("--block-triton", action="store_true")
    parser.add_argument(
        "--large-auto",
        action="store_true",
        help="Also check default routing at 5,000 evaluated rows and 100 columns on CUDA",
    )
    parser.add_argument(
        "--large-shape-auto",
        action="store_true",
        help="Check automatic CUDA Shape at 20,000 evaluated rows with two bootstrap draws",
    )
    parser.add_argument("--smoke", action="store_true", help="Small CPU source-checkout smoke")
    args = parser.parse_args()
    if args.without_compiler:
        os.environ["CXX"] = "/compiler-intentionally-unavailable"
        os.environ["CC"] = "/c-compiler-intentionally-unavailable"
        for name in (
            "CUDA_HOME",
            "TABGEN_EVAL_CUDA_INCLUDE",
            "TABGEN_EVAL_CPU_LIBRARY",
            "TABGEN_EVAL_CUDA_LIBRARY",
        ):
            os.environ.pop(name, None)

    class MissingDependency(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            for blocked in (
                "torch" if args.block_torch else None,
                "triton" if args.block_triton else None,
            ):
                if blocked and (fullname == blocked or fullname.startswith(blocked + ".")):
                    raise ModuleNotFoundError(f"Intentionally unavailable: {blocked}", name=blocked)
            return None

    sys.meta_path.insert(0, MissingDependency())
    import numpy as np
    import pandas as pd

    import tabgen_eval
    from tabgen_eval import EvalData, Schema, evaluate
    from tabgen_eval._evaluation.estimator import Estimator
    from tabgen_eval._evaluation.paired import evaluate_pair
    from tabgen_eval._options import resolve_options
    from tabgen_eval.metrics import LinearC2ST, Shape

    package_path = Path(tabgen_eval.__file__).resolve()
    if args.require_installed:
        assert "site-packages" in package_path.parts, package_path
    rng = np.random.default_rng(1561)
    frames = []
    for shift in (0.0, 0.2):
        frames.append(
            pd.DataFrame(
                {
                    "x": rng.normal(shift, 1, 80 if args.smoke else 440),
                    "y": rng.normal(size=80 if args.smoke else 440),
                    "c": rng.choice(["a", "b", "c"], 80 if args.smoke else 440),
                }
            )
        )
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    data = EvalData(test=frames[0], generated=frames[1], schema=schema)
    metrics = [
        "shape",
        "trend",
        "cond_shape",
        "cond_trend",
        "recovery",
        "precision",
        "recall",
        "density",
        "coverage",
        "linear_c2st",
        "concentration",
    ]
    results = []
    started = time.perf_counter()
    if not args.block_torch:
        for bootstrap in (0, 2):
            report = evaluate(
                data,
                metrics=metrics,
                device=args.device,
                # Import blockers are process-local; normal installations
                # exercise the multiworker path, dependency simulations
                # stay in the process where the dependency is blocked.
                n_jobs=1 if args.block_torch or args.block_triton else 2,
                seed=1562,
                uncertainty={"mode": "both", "n_replicates": bootstrap} if bootstrap else None,
            )
            payload = json.loads(report.to_json(include_replicates=True))
            assert len(report.results) == len(metrics)
            assert not report.to_dataframe().empty
            for metric, stat in report.iter_stats():
                if "NOT_APPLICABLE" in stat.flag_codes:
                    continue
                assert np.isfinite(stat.value), (metric.name, stat.name, stat.flag_codes)
                # Fixed evaluation always includes the paired real-real point.
                assert np.isfinite(stat.ideal), (metric.name, stat.name, stat.flag_codes)
                if bootstrap:
                    assert stat.generated_distribution.requested_count == bootstrap
                    assert stat.reference_distribution.requested_count == bootstrap
                    assert len(stat.generated_distribution.samples) == bootstrap
                    assert len(stat.reference_distribution.samples) == bootstrap
                else:
                    assert stat.generated_distribution is stat.reference_distribution is None
                    assert not hasattr(stat, "comparison")
            execution = payload["provenance"]["metric_execution"]
            assert all(
                v["implementation"] in {"aten_cpp", "triton", "cuda_cpp"}
                for v in execution.values()
            ), execution
            if args.device.startswith("cuda"):
                assert execution["linear_c2st"]["linear_algebra"] == "cublas_cusolver"
                assert execution["linear_c2st"]["device"].startswith("cuda")
            results.append(
                {
                    "case": "torch",
                    "bootstrap": bootstrap,
                    "execution": execution,
                    "report": payload,
                }
            )

    if not args.block_torch and not args.smoke:
        # Force the scalable native solver: the small general report above
        # otherwise takes direct Newton and cannot validate its packaged ABI.
        # A rank below encoded width exercises residual-diagonal preparation.
        for bootstrap in (0, 2):
            saga_report = evaluate_pair(
                frames[0],
                frames[1],
                schema,
                metrics=[
                    LinearC2ST(
                        device=args.device,
                        solver="sketchysaga",
                        saga_rank=2,
                        ridge=0.03,
                    )
                ],
                options=resolve_options(
                    device=args.device,
                    n_jobs=2,
                    seed=1562,
                    uncertainty={"mode": "both", "n_replicates": bootstrap} if bootstrap else None,
                ),
            )
            payload = json.loads(saga_report.to_json(include_replicates=True))
            diagnostics = payload["provenance"]["classifier_diagnostics"]["linear_c2st"]
            for comparison in ("generated", "real_reference"):
                folds = diagnostics["point"][comparison]["folds"]
                assert folds and all(fold["solver"] == "sketchysaga" for fold in folds), diagnostics
                assert all("converged" in fold and "reason" in fold for fold in folds)
            assert np.isfinite(saga_report["linear_c2st/auc"].value)
            results.append({"case": "torch_sketchysaga", "bootstrap": bootstrap, "report": payload})

    if not args.block_torch:
        holdout = pd.DataFrame(
            {
                "x": rng.normal(size=len(frames[0])),
                "y": rng.normal(size=len(frames[0])),
                "c": rng.choice(["a", "b", "c"], len(frames[0])),
            }
        )
        memory = evaluate(
            EvalData(train=frames[0], test=holdout, generated=frames[1], schema=schema),
            group=["memorization"],
            device="cpu" if args.block_torch else args.device,
            seed=1562,
        )
        for _, stat in memory.iter_stats():
            assert np.isfinite(stat.value), (stat.name, stat.flag_codes)
            assert np.isnan(stat.ideal)
            assert stat.generated_distribution is stat.reference_distribution is None
            assert not hasattr(stat, "comparison")
        results.append({"case": "memorization", "report": json.loads(memory.to_json())})

    if args.large_auto:
        if not args.device.startswith("cuda") or args.block_torch or args.block_triton:
            parser.error("--large-auto requires an available explicit CUDA device")
        numbers = [f"n{i}" for i in range(50)]
        categories = [f"c{i}" for i in range(50)]
        wide = []
        for seed in (810, 811):
            random = np.random.default_rng(seed)
            table = pd.DataFrame(random.normal(size=(10000, 50)), columns=numbers)
            for column in categories:
                table[column] = random.integers(10, size=len(table)).astype(str)
            wide.append(table)
        report = evaluate(
            EvalData(
                test=wide[0],
                generated=wide[1],
                schema=Schema(numeric=numbers, categorical=categories),
            ),
            metrics=metrics,
            n_jobs=2,
            seed=1562,
        )
        payload = json.loads(report.to_json())
        execution = payload["provenance"]["metric_execution"]
        for name, info in execution.items():
            assert info["implementation"] in {"aten_cpp", "triton", "cuda_cpp"}, (name, info)
            assert info["device"].startswith("cpu" if name == "shape" else "cuda"), (name, info)
        for metric, stat in report.iter_stats():
            if "NOT_APPLICABLE" not in stat.flag_codes:
                assert np.isfinite(stat.value), (metric.name, stat.name, stat.flag_codes)
        results.append(
            {"case": "auto_large", "bootstrap": 0, "execution": execution, "report": payload}
        )

    if args.large_shape_auto:
        if not args.device.startswith("cuda") or args.block_torch or args.block_triton:
            parser.error("--large-shape-auto requires an available explicit CUDA device")
        numbers, categories = [f"n{i}" for i in range(50)], [f"c{i}" for i in range(50)]
        wide = []
        for seed in (810, 811):
            random = np.random.default_rng(seed)
            table = pd.DataFrame(random.normal(size=(40000, 50)), columns=numbers)
            for column in categories:
                table[column] = random.integers(10, size=len(table)).astype(str)
            wide.append(table)
        report = evaluate(
            EvalData(
                test=wide[0],
                generated=wide[1],
                schema=Schema(numeric=numbers, categorical=categories),
            ),
            metrics=["shape"],
            n_jobs=2,
            seed=1562,
            uncertainty={"mode": "both", "n_replicates": 2},
        )
        payload = json.loads(report.to_json(include_replicates=True))
        execution = payload["provenance"]["metric_execution"]
        assert execution["shape"]["device"].startswith("cuda"), execution
        assert execution["shape"]["implementation"] == "triton"
        assert report["shape/shape"].n_eff == 20000
        assert np.isfinite(report["shape/shape"].value)
        results.append(
            {
                "case": "auto_large_shape",
                "bootstrap": 2,
                "execution": execution,
                "report": payload,
            }
        )

    class CustomShape(Shape):
        def compute(self, reference, candidate, schema, *, rng):
            return {"shape": 0.125}

    custom = Estimator(CustomShape(), resolve_options()).run(*frames, schema)
    assert custom.stats["shape"].value == 0.125
    cuda_dependency_fallback = "not_applicable"
    if args.block_torch:
        try:
            evaluate(
                data,
                metrics=["linear_c2st"],
                device="cpu",
            )
        except ImportError as error:
            assert "torch" in str(error).lower(), str(error)
        else:
            raise AssertionError("Shared classifier preprocessing must require Torch")
        try:
            evaluate(
                data,
                metrics=["shape"],
                device="cpu",
            )
        except ImportError:
            pass
        else:
            raise AssertionError("Automatic built-in evaluation must reject missing Torch")
        torch_version = None
        cuda_version = None
    else:
        import torch

        torch_version, cuda_version = torch.__version__, torch.version.cuda
        if args.block_triton and torch.cuda.is_available():
            from tabgen_eval.execution.runtime.dispatch import context_for
            from tabgen_eval.metrics import PRDC

            # Select a workload above the CUDA threshold without computing its
            # distance matrix. The absent optional dependency must route auto
            # to CPU and make an explicit CUDA request fail clearly.
            large = [pd.concat([frame] * 14, ignore_index=True) for frame in frames]
            context = context_for(PRDC(), *large, schema)
            assert str(context.inputs.device) == "cpu"
            assert context.routes["prdc"] == "cuda_or_triton_unavailable"
            try:
                context_for(PRDC(device="cuda:0"), *large, schema)
            except ImportError:
                pass
            else:
                raise AssertionError("Explicit CUDA must reject missing Triton")
            cuda_dependency_fallback = "passed"
        if args.device == "cpu":
            assert not any(
                name.startswith("tabgen_eval.execution.torch.cuda.triton.") for name in sys.modules
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "package": str(package_path),
                "python": platform.python_version(),
                "torch": torch_version,
                "cuda": cuda_version,
                "device": args.device,
                "without_compiler": args.without_compiler,
                "disabled_compilers": ["CC", "CXX"] if args.without_compiler else [],
                "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
                "blocked_torch": args.block_torch,
                "classifier_missing_torch_rejection": "passed" if args.block_torch else None,
                "blocked_triton": args.block_triton,
                "custom_metric": "passed",
                "cuda_missing_triton_fallback": cuda_dependency_fallback,
                "elapsed_seconds": time.perf_counter() - started,
                "reports": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Installation checks passed: {args.out}")


if __name__ == "__main__":
    main()
