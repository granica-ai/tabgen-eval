"""Execution boundary, owned tensor storage and unchanged statistical protocols."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval._evaluation.estimator import Estimator
from tabgen_eval.execution.runtime.context import UnsupportedExecutionError
from tabgen_eval.execution.runtime.dispatch import ExecutionSession, context_for
from tabgen_eval.metrics import DCR, PRDC, Conditional, LinearC2ST, Shape, Trend
from tabgen_eval.preprocessing.columns import PreparedColumns
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def cpu_context(*frames):
    pytest.importorskip("torch")
    return context_for(Shape(device="cpu"), *frames)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def frames():
    rng = np.random.default_rng(812)
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    r = pd.DataFrame(
        {
            "x": rng.normal(size=260),
            "y": rng.normal(size=260),
            "c": rng.choice(["a", "b", "nan"], 260),
        }
    )
    s = pd.DataFrame(
        {
            "x": rng.normal(size=240),
            "y": rng.normal(size=240),
            "c": rng.choice(["a", "b", "unknown"], 240),
        }
    )
    r.loc[::17, "x"] = np.nan
    s.loc[::23, "c"] = None
    return r, s, schema


def test_host_preparation_owns_data_and_preserves_shared_identities():
    r = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan], "c": ["shared", "nan", None], "ignored": [1.0, np.nan, 3.0]}
    )
    s = pd.DataFrame({"x": [3.0, 4.0], "c": ["shared", "new"], "ignored": [1.0, 2.0]})
    columns = PreparedColumns.prepare(r, s, Schema(numeric=["x"], categorical=["c"]))
    assert columns.reference.categorical[0, 0] == columns.candidate.categorical[0, 0]
    assert columns.reference.categorical[0, 1] >= 0
    assert columns.reference.categorical[0, 1] != columns.reference.categorical[0, 0]
    assert columns.reference.categorical[0, 2] == -1
    assert columns.category_mismatch
    # Missingness in an excluded source column cannot remove a record.
    np.testing.assert_array_equal(columns.reference.complete, [True, True, False])
    r.loc[0, "x"], r.loc[0, "c"] = 100.0, "changed"
    assert columns.reference.numeric[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        columns.reference.numeric[0, 0] = 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cuda:-1"},
        {"device": "mps"},
        {"dtype": "float16"},
    ],
)
def test_execution_options_are_validated_without_torch(kwargs):
    with pytest.raises(ValueError):
        make_options(**kwargs)
    with pytest.raises(ValueError):
        PRDC(**kwargs)


@pytest.mark.parametrize("value", [0, True, 1.5])
def test_private_metric_workspace_is_validated_without_torch(value):
    with pytest.raises(ValueError, match="gpu_workspace_mb"):
        PRDC(gpu_workspace_mb=value)


def test_custom_execution_does_not_import_torch():
    code = """
import sys
import pandas as pd
from tabgen_eval import Schema
from tests.support.options import make_options
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import memorization_report as evaluate_memorization
from tabgen_eval._evaluation.estimator import Estimator
from tabgen_eval.metrics import Shape
class CustomShape(Shape):
    def compute(self, reference, candidate, schema, *, rng):
        return {'shape': 0.125}
df = pd.DataFrame({'x': range(200)})
schema = Schema(numeric=['x'])
assert 'torch' not in sys.modules
Estimator(CustomShape(), make_options(uncertainty="none")).run(df, df, schema)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=30)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tensor_storage_precision_ownership_and_gather(torch, dtype):
    r = pd.DataFrame({"x": [1.0 + 2**-40, 2.0, np.nan], "c": ["a", "nan", None]})
    s = pd.DataFrame({"x": [1.0, 3.0], "c": ["a", "new"]})
    columns = PreparedColumns.prepare(r, s, Schema(numeric=["x"], categorical=["c"]))
    inputs = columns.to_tensors(dtype=dtype)
    table = inputs.reference
    assert table.numeric.dtype == getattr(torch, dtype)
    assert table.categorical.dtype == torch.int64
    assert table.complete.dtype == torch.bool
    assert table.numeric.shape == (3, 1)
    expected = columns.reference.numeric.T.astype(dtype)
    np.testing.assert_array_equal(table.numeric.numpy(), expected)
    assert table.numeric[0, 0].item() == (1.0 if dtype == "float32" else 1.0 + 2**-40)
    draw = table.gather(np.array([1, 1, 0, 2]))
    np.testing.assert_array_equal(draw.numeric.numpy(), expected[[1, 1, 0, 2]])
    np.testing.assert_array_equal(draw.complete.numpy(), [True, True, True, False])
    draw.numeric[0, 0] = 9.0
    assert table.numeric[1, 0] == 2.0
    table.numeric[0, 0] = 7.0
    assert columns.reference.numeric[0, 0] == 1.0 + 2**-40
    assert not table.numeric.requires_grad
    assert table.numeric.is_contiguous()
    clone = inputs.to(device="cpu", dtype=dtype)
    clone.reference.numeric[1, 0] = 11.0
    assert table.numeric[1, 0] == 2.0
    assert table.gather(np.empty(0, dtype=np.int64)).numeric.shape == (0, 1)


@pytest.mark.parametrize("schema", [Schema(numeric=["x"]), Schema(categorical=["x"]), Schema()])
def test_empty_tensor_blocks(torch, schema):
    df = pd.DataFrame({"x": pd.Series([], dtype=float)})
    pair = PreparedColumns.prepare(df, df, schema).to_tensors()
    assert pair.reference.numeric.shape == (0, len(schema.numeric))
    assert pair.reference.categorical.shape == (0, len(schema.categorical))
    assert pair.reference.complete.shape == (0,)
    assert pair.nbytes == 0


@pytest.mark.parametrize(
    "bad", [[-1], [3], [0.5], [[0]], [True], np.array([2**64 - 1], dtype=np.uint64)]
)
def test_tensor_gather_rejects_invalid_indices(torch, bad):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    table = PreparedColumns.prepare(df, df, Schema(numeric=["x"])).to_tensors().reference
    with pytest.raises(ValueError, match="draw"):
        table.gather(bad)


def test_missing_and_broken_torch_dependencies_are_distinguished(monkeypatch, frames):
    from tabgen_eval.execution.torch import tensors

    original = tensors.importlib.import_module

    def missing(name):
        if name == "torch":
            raise ModuleNotFoundError(name="torch")
        return original(name)

    monkeypatch.setattr(tensors.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="torch"):
        context_for(Shape(), *frames)
    with pytest.raises(ImportError, match="tabgen-eval\\[torch\\]"):
        context_for(Shape(device="cpu"), *frames)
    for kwargs in [{"device": "cpu"}, {"dtype": "float32"}, {"device": "cuda:0"}]:
        with pytest.raises(ImportError, match="tabgen-eval\\[torch\\]"):
            context_for(Shape(**kwargs), *frames)

    def broken(name):
        if name == "torch":
            raise ImportError("undefined symbol")
        return original(name)

    monkeypatch.setattr(tensors.importlib, "import_module", broken)
    with pytest.raises(ImportError, match="undefined symbol"):
        context_for(Shape(), *frames)
    with pytest.raises(ImportError, match="undefined symbol"):
        context_for(Shape(device="cpu"), *frames)

    def dependency_missing(name):
        if name == "torch":
            raise ModuleNotFoundError(name="torch._C")
        return original(name)

    monkeypatch.setattr(tensors.importlib, "import_module", dependency_missing)
    with pytest.raises(ModuleNotFoundError) as error:
        context_for(Shape(device="cpu"), *frames)
    assert error.value.name == "torch._C"


def test_default_frontend_is_torch_and_small_reports_use_cpu(torch, frames):
    metrics = ["shape", "trend", "conditional_fidelity", "prdc", "linear_c2st"]
    report = evaluate(
        *frames, metrics=metrics, options=make_options(uncertainty="both", n_replicates=2, n_jobs=4)
    )
    assert {
        name: info["implementation"] for name, info in report.provenance["metric_execution"].items()
    } == dict.fromkeys(metrics, "aten_cpp")
    assert all(v["device"] == "cpu" for v in report.provenance["metric_execution"].values())
    memory = evaluate_memorization(*frames, options=make_options(n_jobs=4))
    assert memory.provenance["metric_execution"]["dcr"]["implementation"] == "aten_cpp"
    assert memory.provenance["metric_execution"]["dcr"]["device"] == "cpu"
    assert (
        report.provenance["metric_execution"]["conditional_fidelity"]["routing_reason"]
        == "small_workload_cpu"
    )


def test_numeric_dcr_uses_native_cpu_for_auto_and_explicit_options(torch):
    schema = Schema(numeric=["x"])
    r = pd.DataFrame({"x": np.arange(200, dtype=float)})
    memory = evaluate_memorization(r, r, schema)
    assert memory.provenance["metric_execution"]["dcr"]["implementation"] == "aten_cpp"
    for kwargs in [{"device": "cpu"}, {"dtype": "float32"}]:
        metric = DCR(**kwargs)
        assert context_for(metric, r, r, schema).inputs.device.type == "cpu"


def test_generated_size_drives_default_routing_instead_of_source_rows(torch):
    rng = np.random.default_rng(3411)
    columns = [f"x{i}" for i in range(100)]
    frame = pd.DataFrame(rng.normal(size=(10000, 100)), columns=columns)
    report = evaluate(
        frame,
        frame.iloc[:100],
        Schema(numeric=columns),
        metrics=["conditional_fidelity"],
        options=make_options(uncertainty="none"),
    )
    assert report.provenance["n_eff"] == 100
    info = report.provenance["metric_execution"]["conditional_fidelity"]
    assert info["device"] == "cpu" and info["routing_reason"] == "small_workload_cpu"


def test_large_workload_with_tiny_gpu_budget_routes_cpu_before_cuda_probe(torch, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("known oversized buffers must route CPU before querying CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    rng = np.random.default_rng(3412)
    names = [f"x{i}" for i in range(100)]
    frame = pd.DataFrame(rng.normal(size=(5000, 100)), columns=names)
    context = context_for(Conditional(gpu_workspace_mb=1), frame, frame, Schema(numeric=names))
    assert context.inputs.device.type == "cpu"
    assert context.execution_info_for(Conditional())["routing_reason"] == "gpu_workspace_too_small"


@pytest.mark.parametrize(
    "torch_version,cuda_version,triton_version,supported",
    [
        ("2.11.0+cu130", "13.0", "3.6.0", True),
        ("2.6.0+cu118", "11.8", "3.2.0", True),
        ("2.6.1+cu118", "11.8", "3.2.1", True),
        ("2.6.0+cu124", "12.4", "3.2.0", True),
        ("2.6.0+cu118", "11.8", "3.6.0", True),
        ("2.11.0+cu130", "13.0", "3.2.0", True),
        ("2.60.0", "11.8", "3.2.0", True),
        ("2.6.0+cu118", "11.8", "3.20.0", True),
        ("3.0.0", "14.0", "4.0.0", True),
        ("2.5.1", "13.0", "3.6.0", False),
        ("2.11.0", "11.7", "3.6.0", False),
        ("2.11.0", "11.10", "3.6.0", True),
        ("2.11.0", "11.8", "3.1.9", False),
        ("2.11.0+cpu", None, "3.6.0", False),
        ("unknown", "13.0", "3.6.0", False),
        ("2.11.0", "unknown", "3.6.0", False),
        ("2.11.0", "13.0", "unknown", False),
        ("2.6.0+cu118", "11.8", None, False),
    ],
)
def test_cuda_minimum_versions(monkeypatch, torch_version, cuda_version, triton_version, supported):
    from tabgen_eval.execution.torch import context

    def installed_version(name):
        assert name == "triton"
        if triton_version is None:
            raise context.PackageNotFoundError(name)
        return triton_version

    monkeypatch.setattr(context, "version", installed_version)
    module = SimpleNamespace(__version__=torch_version, version=SimpleNamespace(cuda=cuda_version))
    assert context.cuda_supported(module) is supported


def test_unsupported_cuda_versions_fall_back_only_for_auto(torch, monkeypatch):
    from tabgen_eval.execution.torch import context as torch_context

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch_context, "find_spec", lambda name: object())
    monkeypatch.setattr(torch_context, "cuda_supported", lambda module: False)
    frame = pd.DataFrame(np.zeros((5000, 100)), columns=[f"x{i}" for i in range(100)])
    schema = Schema(numeric=list(frame))
    metric = Trend()
    context = context_for(metric, frame, frame, schema)
    assert context.inputs.device.type == "cpu"
    assert context.execution_info_for(metric)["routing_reason"] == "cuda_version_below_minimum"
    with pytest.raises(
        UnsupportedExecutionError, match=r"Torch >=2\.6, CUDA >=11\.8 and Triton >=3\.2"
    ):
        context_for(Trend(device="cuda:0"), frame, frame, schema)


def test_mixed_bootstrap_counts_and_cached_residency_influence_pairwise_routing():
    from tabgen_eval.execution.runtime.routing import Workload

    point = Workload(500, 50, 50, (10,) * 50, 1000, 1000, 500)
    draws = Workload(500, 50, 50, (10,) * 50, 1000, 1000, 500, n_bootstrap=2)
    resident = Workload(500, 50, 50, (10,) * 50, 1000, 1000, 500, cuda_resident=True)
    assert not point.cuda_candidate(PRDC())[0]
    assert draws.cuda_candidate(PRDC())[0]
    assert resident.cuda_candidate(DCR())[0]
    # Missing-row reductions shrink the actual pairwise workload independently
    # of a high raw/source row count.
    sparse = Workload(5000, 50, 50, (10,) * 50, 10000, 10000, 100, n_bootstrap=2)
    assert not sparse.cuda_candidate(PRDC())[0]


def test_cuda_residency_matches_device_precision_and_budget(torch, monkeypatch, frames):
    session = ExecutionSession(*frames)
    session._contexts[("torch", "cuda:1", "float64", 0)] = object()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert session._workload(PRDC()).cuda_resident
    assert not session._workload(PRDC(dtype="float32")).cuda_resident
    assert not session._workload(PRDC(gpu_workspace_mb=512)).cuda_resident
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert not session._workload(PRDC()).cuda_resident


@pytest.mark.gpu
def test_default_large_report_routes_all_gpu_metrics_and_retains_cpu_shape(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from tabgen_eval.execution.torch.context import cuda_supported

    if not cuda_supported(torch):
        pytest.skip("outside supported CUDA release matrix")
    rng = np.random.default_rng(3413)
    numeric = [f"n{i}" for i in range(50)]
    categorical = [f"c{i}" for i in range(50)]
    tables = []
    for _ in range(2):
        table = pd.DataFrame(rng.normal(size=(10000, 50)), columns=numeric)
        for column in categorical:
            table[column] = rng.integers(10, size=len(table)).astype(str)
        tables.append(table)
    schema = Schema(numeric=numeric, categorical=categorical)
    metrics = ["shape", "trend", "prdc", "conditional_fidelity", "linear_c2st"]
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    # Auto respects the caller's selected GPU and restores it on exit.
    with torch.cuda.device(device):
        options = make_options(uncertainty="none", n_jobs=4)
        report = evaluate(*tables, schema, metrics=metrics, options=options)
        memory = evaluate_memorization(*tables, schema, options=options)
    other = evaluate(
        *tables,
        schema,
        metrics=metrics,
        options=make_options(device="cpu", uncertainty="none", n_jobs=4),
    )
    expected_memory = evaluate_memorization(
        *tables,
        schema,
        options=make_options(device="cpu", n_jobs=4),
    )
    assert memory.provenance["metric_execution"]["dcr"]["device"] == device
    np.testing.assert_allclose(
        [stat.value for stat in memory.results[0].stats.values()],
        [stat.value for stat in expected_memory.results[0].stats.values()],
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    assert memory.results[0].diagnostics.get("stat_flags", {}) == expected_memory.results[
        0
    ].diagnostics.get("stat_flags", {})
    info = report.provenance["metric_execution"]
    assert info["shape"]["device"] == "cpu"
    assert all(info[name]["device"] == device for name in metrics if name != "shape")
    assert report.provenance["metric_execution"]["linear_c2st"]["implementation"] == "cuda_cpp"
    for result, expected in zip(report.results, other.results, strict=True):
        for name, actual in result.stats.items():
            assert actual.flags == expected.stats[name].flags
            assert actual.value == pytest.approx(
                expected.stats[name].value,
                abs=2e-5 if result.name == "linear_c2st" else 1e-10,
                nan_ok=True,
            )


def test_auto_device_uses_implemented_cpu_without_probing_cuda(torch, monkeypatch, frames):
    def forbidden():
        pytest.fail("CPU auto execution must not probe CUDA before a CUDA kernel exists")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    context = context_for(Shape(), *frames)
    assert str(context.inputs.device) == "cpu"


def test_cpu_fp32_classifier_supported_and_unavailable_cuda_raises(torch, monkeypatch, frames):
    context = context_for(LinearC2ST(device="cpu", dtype="float32"), *frames)
    assert context.inputs.reference.numeric.dtype == torch.float32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        context_for(PRDC(device="cuda:0"), *frames)


def test_context_cache_shares_preparation_and_records_actual_implementation(
    torch, monkeypatch, frames
):
    count = 0
    original = PreparedColumns.prepare.__func__

    def counted(cls, *args):
        nonlocal count
        count += 1
        return original(cls, *args)

    monkeypatch.setattr(PreparedColumns, "prepare", classmethod(counted))
    session = ExecutionSession(*frames)
    tensor = session.context_for(Shape(device="cpu"))
    assert session.context_for(Trend(device="cpu")) is tensor
    assert count == 1
    assert tensor.execution_info_for(Shape()) == {
        "implementation": "aten_cpp",
        "device": "cpu",
        "dtype": "float64",
        "source_pools": 2,
        "persistent_input_bytes": tensor.inputs.nbytes,
    }


@pytest.mark.parametrize("B", [0, 3])
@pytest.mark.parametrize("jobs", [1, 4])
def test_compiled_cpu_preserves_complete_reports_and_every_replicate(torch, frames, B, jobs):
    metrics = ["shape", "conditional_fidelity", "linear_c2st"]
    common = dict(uncertainty="both" if B else "none", n_replicates=max(B, 1), n_jobs=jobs, seed=25)
    a = evaluate(*frames, metrics=metrics, options=make_options(**common))
    b = evaluate(*frames, metrics=metrics, options=make_options(device="cpu", **common))
    for mr, x in a.iter_stats():
        y = b[f"{mr.name}/{x.name}"]
        assert x.flags == y.flags
        for field in [
            "value",
            "ideal",
            "ci_low",
            "ci_high",
            "ideal_ci_low",
            "ideal_ci_high",
            "generated_distribution",
            "reference_distribution",
        ]:
            av, bv = getattr(x, field), getattr(y, field)

            if hasattr(av, "samples"):
                av, bv = av.samples, bv.samples
            if av is None:
                assert bv is None
            else:
                np.testing.assert_allclose(
                    av, bv, rtol=0, atol=2e-4 if mr.name == "linear_c2st" else 1e-12, equal_nan=True
                )
    assert {
        name: info["implementation"] for name, info in b.provenance["metric_execution"].items()
    } == dict.fromkeys(metrics, "aten_cpp")
    assert all(
        info["implementation"] == "aten_cpp" for info in b.provenance["metric_execution"].values()
    )


@pytest.mark.parametrize("metric", [Shape, Trend, Conditional, PRDC, DCR, LinearC2ST])
def test_direct_metric_calls_use_the_same_tensor_dispatch(torch, frames, metric):
    a = metric().compute(*frames, rng=np.random.default_rng(22))
    b = metric(device="cpu").compute(*frames, rng=np.random.default_rng(22))
    if metric in {PRDC, DCR}:
        # Pairwise reductions may differ in their final floating-point summation.
        np.testing.assert_allclose(list(a.values()), list(b.values()), rtol=1e-6, atol=1e-7)
    else:
        np.testing.assert_allclose(
            list(a.values()), list(b.values()), rtol=0, atol=2e-4 if metric is LinearC2ST else 1e-12
        )


def test_explicit_metric_instances_keep_backend_and_precision_provenance(torch, frames):
    report = evaluate(
        *frames,
        metrics=[Shape(), Trend(device="cpu", dtype="float64")],
        options=make_options(
            device="cpu",
            dtype="float32",
            uncertainty="none",
        ),
    )
    info = report.provenance["metric_execution"]
    assert info["shape"]["implementation"] == "aten_cpp"
    assert info["trend"]["implementation"] == "aten_cpp"
    assert info["trend"]["dtype"] == "float64"


def test_cli_passes_tensor_execution_options(torch, frames, tmp_path):
    from tabgen_eval.cli import main

    r, s, schema = frames
    r.to_csv(tmp_path / "real.csv", index=False)
    s.to_csv(tmp_path / "generated.csv", index=False)
    (tmp_path / "schema.json").write_text(json.dumps(schema.to_dict()))
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--holdout",
                str(tmp_path / "real.csv"),
                "--generated",
                str(tmp_path / "generated.csv"),
                "--schema",
                str(tmp_path / "schema.json"),
                "--metrics",
                "shape",
                "--device",
                "cpu",
                "--dtype",
                "float64",
                "--uncertainty",
                "none",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert "gpu_workspace_mb" not in payload["options"]
    execution = payload["provenance"]["metric_execution"]["shape"]
    memory_plan = execution.pop("memory_plan")
    assert memory_plan["policy"] == "available_memory_80_percent"
    assert memory_plan["device"] == "cpu"
    assert memory_plan["task_workspace_bytes"] > 0
    assert execution == {
        "implementation": "aten_cpp",
        "device": "cpu",
        "dtype": "float64",
        "source_pools": 2,
        "persistent_input_bytes": (len(frames[0]) + len(frames[1])) * (2 * 8 + 8 + 1),
    }


@pytest.mark.parametrize("metric", [Shape(), DCR(), LinearC2ST()])
def test_cpu_batch_policy_matches_existing_budgets(torch, frames, metric):
    context = cpu_context(*frames)
    pools = 2
    width = context.classifier_width if metric.name == "linear_c2st" else 3
    for n in [1, 1000, 100000, 1000000]:
        expected = max(
            1, min(64, (8 * 1024**2) // (8 * pools * n), (256 * 1024**2) // (8 * pools * n * width))
        )
        assert context.batch_size(metric, n) == expected


@pytest.fixture
def cuda_device(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Use the idle second device on the profiling host; single-device hosts use zero.
    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_preparation_gather_and_roundtrip(torch, cuda_device, frames, dtype):
    from tabgen_eval.execution.torch.context import TorchContext

    columns = PreparedColumns.prepare(*frames)
    context = TorchContext.from_columns(columns, device=cuda_device, dtype=dtype)
    assert str(context.inputs.device) == cuda_device
    assert context.inputs.dtype == dtype
    draw = context.gather(np.array([3, 3, 0, 19]))
    np.testing.assert_array_equal(
        draw.numeric.cpu().numpy(), columns.reference.numeric.T[[3, 3, 0, 19]].astype(dtype)
    )
    np.testing.assert_array_equal(
        draw.categorical.cpu().numpy(), columns.reference.categorical.T[[3, 3, 0, 19]]
    )
    assert not draw.numeric.requires_grad
    roundtrip = context.inputs.to(device="cpu", dtype=dtype)
    np.testing.assert_array_equal(
        roundtrip.reference.numeric.numpy(), columns.reference.numeric.T.astype(dtype)
    )

    class UnsupportedMetric(Shape):
        pass

    with pytest.raises(UnsupportedExecutionError, match="No torch metric implementation"):
        context.compute(UnsupportedMetric(), np.array([[0, 1]]), np.array([[0, 1]]))


@pytest.mark.gpu
def test_explicit_cuda_metric_is_not_silently_run_on_cpu(torch, cuda_device, frames):
    report = evaluate(
        *frames,
        metrics=["shape"],
        options=make_options(device=cuda_device, uncertainty="none"),
    )
    assert report.provenance["metric_execution"]["shape"]["device"] == cuda_device
    assert report.provenance["metric_execution"]["shape"]["implementation"] == "triton"


@pytest.mark.gpu
def test_gpu_budget_bounds_gathers_and_leaves_global_precision_unchanged(
    torch, cuda_device, frames, monkeypatch
):
    from tabgen_eval.execution.torch.context import TorchContext

    precision = torch.get_float32_matmul_precision()
    context = TorchContext.from_columns(
        PreparedColumns.prepare(*frames), device=cuda_device, dtype="float32", gpu_workspace_mb=1
    )
    assert context.batch_size(PRDC(), 100000) < 64
    with pytest.raises(MemoryError, match="workspace allowance"):
        context.compute(
            PRDC(), np.zeros((1, 1000000), dtype=np.int64), np.zeros((1, 1000000), dtype=np.int64)
        )
    with pytest.raises(MemoryError, match="workspace allowance"):
        context.gather(np.zeros(1000000, dtype=np.int64))

    # Index arrays alone fit, but their filtering scratch does not. Reject before
    # uploading indices, rather than only checking after complete-row filtering.
    def forbidden(*args, **kwargs):
        pytest.fail("oversized index filtering must be rejected before device upload")

    monkeypatch.setattr(type(context.reference), "row_indices", forbidden)
    with pytest.raises(MemoryError, match="index filtering"):
        context.compute(
            PRDC(), np.zeros((1, 40000), dtype=np.int64), np.zeros((1, 40000), dtype=np.int64)
        )
    assert torch.get_float32_matmul_precision() == precision


@pytest.mark.gpu
def test_cuda_invalid_device_and_integer_draws(torch, cuda_device, frames):
    from tabgen_eval.execution.torch.tensors import resolve_device

    with pytest.raises(ValueError, match="index"):
        resolve_device(f"cuda:{torch.cuda.device_count()}")
    inputs = PreparedColumns.prepare(*frames).to_tensors(device=cuda_device)
    for bad in [torch.tensor([-1], device=cuda_device), torch.tensor([0.5], device=cuda_device)]:
        with pytest.raises(ValueError, match="draw"):
            inputs.reference.gather(bad)
    drawn = inputs.reference.gather(torch.tensor([2, 2, 0], device=cuda_device))
    assert drawn.n_rows == 3


def test_estimator_calls_context_batch_policy_without_changing_draws(torch, frames):
    context = cpu_context(*frames)

    class SmallBatches:
        def __init__(self):
            self.requests = []

        def __getattr__(self, name):
            return getattr(context, name)

        def batch_size(self, metric, n):
            self.requests.append((metric.name, n))
            return 2

    small = SmallBatches()
    cfg = make_options(device="cpu", uncertainty="both", n_replicates=9)
    a = Estimator(Shape(), cfg, context=context).run(*frames)
    b = Estimator(Shape(), cfg, context=small).run(*frames)
    assert small.requests
    for name in a.stats:
        np.testing.assert_array_equal(
            a.stats[name].generated_distribution.samples,
            b.stats[name].generated_distribution.samples,
        )
        np.testing.assert_array_equal(
            a.stats[name].reference_distribution.samples,
            b.stats[name].reference_distribution.samples,
        )


def test_custom_metric_instance_and_subclass_preserve_compute_contract(frames):
    class CustomShape(Shape):
        def compute(self, reference, candidate, schema, *, rng):
            return {"shape": 0.125}

    metrics = [
        CustomShape(),
        Shape(compute=lambda reference, candidate, schema, *, rng: {"shape": 0.375}),
    ]
    for metric, expected in zip(metrics, [0.125, 0.375], strict=True):
        outcome = Estimator(metric, make_options(uncertainty="both", n_replicates=2)).run(*frames)
        assert outcome.stats["shape"].value == expected
        assert context_for(metric, *frames) is None
