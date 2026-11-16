"""Independent precision oracles and end-to-end tensor distance execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import DCR, PRDC
from tests.support.distances import distance_matrices, memorization
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def oracle_pair(reference, queries, schema, dtype):
    return distance_matrices(reference, queries, schema, dtype)


def oracle_prdc(real, fake, schema, dtype, k):
    (distances, (nr, ns)) = oracle_pair(real, [fake], schema, dtype)
    rr = np.partition(distances(0, 0), k, axis=1)[:, k]
    sr = np.partition(distances(1, 1), k, axis=1)[:, k]
    cross = distances(0, 1)
    inside = cross < rr[:, None]
    return {
        "precision": inside.any(axis=0).mean(),
        "recall": (cross < sr[None, :]).any(axis=1).mean(),
        "density": inside.sum() / (k * ns),
        "coverage": inside.any(axis=1).mean(),
    }


def frames(n=57, wide=False):
    rng = np.random.default_rng(772)
    r = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "constant": np.ones(n) * 7,
            "c": [f"level-{i}" for i in range(n)] if wide else rng.choice(["a", "b", "nan"], n),
        }
    )
    s = pd.DataFrame(
        {
            "x": rng.normal(0.2, size=n + 5),
            "constant": np.ones(n + 5) * 7,
            "c": [f"unseen-{i}" for i in range(n + 5)]
            if wide
            else rng.choice(["a", "b", "unknown"], n + 5),
        }
    )
    r.loc[3] = r.iloc[2]
    s.loc[4] = s.iloc[2]
    r.loc[::19, "x"] = np.nan
    s.loc[::23, "c"] = None
    return (r, s, Schema(numeric=["x", "constant"], categorical=["c"]))


@pytest.mark.parametrize("wide", [False, True])
def test_prdc_matches_independent_onehot_oracle(torch, device, dtype, wide):
    (r, s, schema) = frames(wide=wide)
    expected = oracle_prdc(r, s, schema, dtype, 3)
    actual = PRDC(device=device, dtype=dtype, nearest_k=3).compute(r, s, schema, rng=None)
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-12, rtol=0)


def test_dcr_matches_independent_medians_and_preserves_input_pools(torch, device, dtype):
    (r, s, schema) = frames(n=70, wide=True)
    a = r.iloc[:35]
    (matrices, counts) = oracle_pair(a, [s], schema, dtype)
    expected = np.median(np.sqrt(matrices(1, 0).min(axis=1)))
    context = context_for(DCR(device=device, dtype=dtype), r, s, schema)
    before = context.reference.numeric.clone()
    outcome = context.compute_memorization_group(
        [DCR()], np.arange(35)[None], np.arange(len(s))[None]
    )[0]["dcr"]
    tolerance = 1e-06 if dtype == "float32" else 1e-12
    assert outcome.values["median"] == pytest.approx(expected, rel=tolerance, abs=tolerance)
    assert outcome.diagnostics["usable_rows"] == {
        "training": counts[0],
        "generated": counts[1],
        "holdout": None,
    }
    assert outcome.diagnostics["holdout_median"] is None
    assert np.isnan(outcome.values["ratio"])
    torch.testing.assert_close(context.reference.numeric, before, rtol=0, atol=0, equal_nan=True)


def test_even_euclidean_median_and_zero_baseline(torch, device, dtype):
    r = pd.DataFrame({"x": [0.0, 8.0]})
    g = pd.DataFrame({"x": [16.0, 32.0]})
    schema = Schema(numeric=["x"])
    actual = memorization(r, g, schema, device, dtype, holdout=pd.DataFrame({"x": [16.0]}))
    assert actual.values["median"] == 2.0
    assert actual.diagnostics["holdout_median"] == 1.0
    assert actual.values["ratio"] == 2.0
    actual = memorization(r * 0, g * 0, schema, device, dtype, holdout=r * 0)
    assert actual.values["median"] == 0.0 and np.isnan(actual.values["ratio"])


def test_strict_category_boundaries_and_unknown_vs_unknown(torch, device, dtype):
    r = pd.DataFrame({"c": list("abcdefgh")})
    schema = Schema(categorical=["c"])
    metric = PRDC(device=device, dtype=dtype, nearest_k=2)
    assert metric.compute(r, r, schema, rng=None) == {
        "precision": 1.0,
        "recall": 1.0,
        "density": 0.5,
        "coverage": 1.0,
    }
    g = pd.DataFrame({"c": ["unseen-a", "unseen-b"] * 4})
    assert metric.compute(r, g, schema, rng=None) == {
        "precision": 1.0,
        "recall": 0.0,
        "density": 4.0,
        "coverage": 1.0,
    }


def test_streaming_radii_keep_duplicate_multiplicities_across_tiles(torch, device, dtype):
    from tabgen_eval.execution.torch.distance.geometry import (
        DistanceWorkspace,
        ReferenceGeometry,
        kth_radii,
    )

    r = pd.DataFrame({"x": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    schema = Schema(numeric=["x"])
    context = context_for(PRDC(device=device, dtype=dtype), r, r, schema)
    sample = context.gather(np.arange(len(r)))
    encoded = ReferenceGeometry(sample).transform(sample, reference=True)
    workspace = DistanceWorkspace(context, [len(r), len(r)], [len(r), len(r)], k=2)
    workspace.scratch_bytes = 512
    actual = kth_radii(encoded, 2, workspace).cpu().numpy()
    direct = (r.x.to_numpy(dtype=dtype)[:, None] / 6 - r.x.to_numpy(dtype=dtype)[None, :] / 6) ** 2
    expected = np.partition(direct, 2, axis=1)[:, 2]
    np.testing.assert_allclose(
        actual, expected, rtol=1e-06 if dtype == "float32" else 1e-12, atol=0
    )
    assert (actual[:3] == 0).all()


@pytest.mark.parametrize(
    "op_name", ["squared_distance_tile", "nearest_squared_tile", "prdc_cross_tile"]
)
def test_registered_operator_schema_and_fake_tensor_contract(torch, device, dtype, op_name):
    from tabgen_eval.execution.torch.distance import ops as distance_ops

    a = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]], device=device, dtype=getattr(torch, dtype)
    )
    b = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device, dtype=getattr(torch, dtype))
    ca = torch.tensor([[0], [-1], [2]], device=device)
    cb = torch.tensor([[0], [-1]], device=device)
    args = (a, b, ca, cb)
    if op_name == "prdc_cross_tile":
        args += (
            torch.ones(3, device=device, dtype=a.dtype),
            torch.ones(2, device=device, dtype=a.dtype),
        )
    assert all(
        v == "SUCCESS"
        for v in torch.library.opcheck(
            getattr(distance_ops, op_name), args, test_utils=("test_schema", "test_faketensor")
        ).values()
    )


def test_direct_distance_avoids_cancellation_without_refinement(torch, device, dtype):
    from tabgen_eval.execution.torch.distance.ops import squared_distance_tile

    kind = getattr(torch, dtype)
    a = torch.tensor(
        [[1000000.0, 1000000.0], [1000000.0 + 0.125, 1000000.0]], dtype=kind, device=device
    )
    cats = torch.empty((2, 0), dtype=torch.int64, device=device)
    actual = squared_distance_tile(a, a, cats, cats).cpu().numpy()
    np.testing.assert_array_equal(actual, [[0.0, 0.015625], [0.015625, 0.0]])
    assert str(torch.ops.tabgen_eval.squared_distance_tile) == "tabgen_eval.squared_distance_tile"


def test_fp64_geometry_never_round_trips_through_fp32(torch, device, dtype):
    r = pd.DataFrame({"x": 1000000000000.0 + np.array([0.0, 0.5, 1.0])})
    g = pd.DataFrame({"x": [1000000000000.0 + 1.5]})
    outcome = memorization(r, g, Schema(numeric=["x"]), device, dtype, holdout=g)
    if dtype == "float64":
        assert outcome.diagnostics["holdout_median"] == 0.5
    else:
        assert outcome.values["median"] == 0.0
        assert np.isnan(outcome.values["ratio"])
        assert outcome.stat_flags == {"ratio": ["ZERO_BASELINE"]}


def test_operator_precision_survives_ambient_autocast(torch, device, dtype):
    from tabgen_eval.execution.torch.distance.ops import squared_distance_tile

    kind = getattr(torch, dtype)
    a = torch.tensor([[0.125, 0.75], [0.625, 1.5]], device=device, dtype=kind)
    cats = torch.empty((2, 0), dtype=torch.int64, device=device)
    expected = squared_distance_tile(a, a, cats, cats)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu"):
        actual = squared_distance_tile(a, a, cats, cats)
    assert actual.dtype == kind
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_primitive_empty_axes_and_strided_radii_validation(torch, device, dtype):
    from tabgen_eval.execution.torch.distance.ops import (
        nearest_squared_tile,
        prdc_cross_tile,
        squared_distance_tile,
    )

    kind = getattr(torch, dtype)
    a = torch.ones((3, 1), device=device, dtype=kind)
    cats = torch.empty((3, 0), dtype=torch.int64, device=device)
    for q, r in [(0, 3), (3, 0), (0, 0)]:
        assert squared_distance_tile(a[:q], a[:r], cats[:q], cats[:r]).shape == (q, r)
        nearest = nearest_squared_tile(a[:q], a[:r], cats[:q], cats[:r])
        assert nearest.shape == (q,)
        if not r:
            assert torch.isinf(nearest).all()
        (count, recall, cover) = prdc_cross_tile(
            a[:q], a[:r], cats[:q], cats[:r], a.new_ones(q), a.new_ones(r)
        )
        assert count.shape == (r,) and recall.shape == cover.shape == (q,)
        assert not count.any() and (not recall.any()) and (not cover.any())
    strided = a.new_ones(6)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        prdc_cross_tile(a, a, cats, cats, strided, a.new_ones(3))


def test_boundary_validation_empty_batches_and_noncomputable_draws(torch, device, dtype):
    (r, s, schema) = frames()
    context = context_for(PRDC(device=device, dtype=dtype), r, s, schema)
    empty = np.empty((0, 5), dtype=np.int64)
    assert all(v.shape == (0,) for v in context.compute(PRDC(), empty, empty).values())
    assert context.compute_memorization_group([DCR()], empty, empty) == []
    for invalid in [np.array([[-1]]), np.array([[len(r)]]), np.array([[1.5]])]:
        with pytest.raises(ValueError, match="draw"):
            context.compute(PRDC(), invalid, np.array([[0]]))
        with pytest.raises(ValueError, match="draw"):
            context.compute_memorization_group([DCR()], invalid, np.array([[0]]))
    with pytest.raises(ValueError, match="batch lengths"):
        context.compute(PRDC(), np.array([[0], [1]]), np.array([[0]]))
    nan_scores = context.compute(PRDC(), np.array([[0, 0]]), np.array([[0, 1]]))
    assert all(np.isnan(v).all() for v in nan_scores.values())
    result = context.compute_memorization_group([DCR()], np.array([[1, 2]]), np.array([[0, 0]]))[0][
        "dcr"
    ]
    assert np.isnan(result.values["median"])
    assert result.diagnostics["holdout_median"] is None


def test_missing_candidate_preserves_finite_baseline(torch, device, dtype):
    r = pd.DataFrame({"x": [0.0, 8.0]})
    g = pd.DataFrame({"x": [np.nan]})
    result = memorization(
        r, g, Schema(numeric=["x"]), device, dtype, holdout=pd.DataFrame({"x": [16.0]})
    )
    assert np.isnan(result.values["median"]) and np.isnan(result.values["ratio"])
    assert result.diagnostics["holdout_median"] == 1.0


@pytest.mark.parametrize("B", [0, 2, 67])
def test_cpu_cuda_reports_preserve_sampling_and_pairing(torch, device, dtype, B):
    (r, s, schema) = frames(n=260)
    options = dict(
        dtype=dtype,
        uncertainty="both" if B else "none",
        n_replicates=max(B, 1),
        seed=37,
    )
    expected = evaluate(
        r, s, schema, metrics=["prdc"], options=make_options(device="cpu", **options)
    )
    actual = evaluate(
        r, s, schema, metrics=["prdc"], options=make_options(device=device, **options)
    )
    tolerance = 2e-06 if dtype == "float32" else 1e-12
    for a, b in zip(expected.results, actual.results, strict=True):
        for name in a.stats:
            (x, y) = (a.stats[name], b.stats[name])
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
                (av, bv) = (getattr(x, field), getattr(y, field))
                if hasattr(av, "samples"):
                    (av, bv) = (av.samples, bv.samples)
                if av is None:
                    assert bv is None
                else:
                    np.testing.assert_allclose(
                        av, bv, rtol=tolerance, atol=tolerance, equal_nan=True
                    )
    info = actual.provenance["metric_execution"]
    assert all(v["dtype"] == dtype and v["device"] == device for v in info.values())
    assert {
        name: info["implementation"] for name, info in actual.provenance["metric_execution"].items()
    } == dict.fromkeys(["prdc"], "triton" if device.startswith("cuda") else "aten_cpp")
    memory_config = make_options(device=device, **options)
    memory = evaluate_memorization(r, s, schema, holdout=s.iloc[1::3], options=memory_config)
    expected_memory = evaluate_memorization(
        r, s, schema, holdout=s.iloc[1::3], options=make_options(device="cpu", **options)
    )
    np.testing.assert_allclose(
        [stat.value for stat in memory.results[0].stats.values()],
        [stat.value for stat in expected_memory.results[0].stats.values()],
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    assert memory.results[0].diagnostics.get("stat_flags", {}) == expected_memory.results[
        0
    ].diagnostics.get("stat_flags", {})
    assert memory.provenance["metric_execution"]["dcr"]["device"] == device
    assert memory.provenance["metric_execution"]["dcr"]["dtype"] == dtype
    assert memory.provenance["metric_execution"]["dcr"]["implementation"] == (
        "triton" if device.startswith("cuda") else "aten_cpp"
    )
    features = list(schema.numeric) + list(schema.categorical)
    n_training = len(r.dropna(subset=features))
    n_generated = len(s.dropna(subset=features))
    assert memory.provenance["n_selected_training"] == n_training
    assert memory.provenance["n_selected_generated"] == n_generated
    assert memory.provenance["n_training"] == len(r)
    assert memory.provenance["n_generated"] == len(s)
    pools = memory.results[0].diagnostics["row_usage"]["pools"]
    assert pools["train"]["retained"] == n_training
    assert pools["generated"]["retained"] == n_generated
    assert pools["test"]["retained"] == len(s.iloc[1::3].dropna(subset=features))
    assert (
        memory["dcr/ratio"].value
        == memory["dcr/median"].value / memory.results[0].diagnostics["holdout_median"]
    )
    point = evaluate_memorization(
        r,
        s,
        schema,
        holdout=s.iloc[1::3],
        options=make_options(device=device, **{**options, "uncertainty": "none"}),
    )
    np.testing.assert_equal(
        [stat.value for stat in point.results[0].stats.values()],
        [stat.value for stat in memory.results[0].stats.values()],
    )
    assert point.results[0].diagnostics == memory.results[0].diagnostics
    assert not hasattr(memory, "ideal_replicates")


def test_cpu_distance_execution_does_not_import_triton(torch, monkeypatch, dtype):
    from tabgen_eval.execution.torch.distance import ops as distance_ops

    def forbidden():
        pytest.fail("CPU operators must not load CUDA kernels")

    monkeypatch.setattr(distance_ops, "cuda_kernels", forbidden)
    (r, s, schema) = frames()
    PRDC(device="cpu", dtype=dtype).compute(r, s, schema, rng=None)


def test_cuda_dependency_errors_and_automatic_cpu_fallback(torch, monkeypatch, dtype):
    from tabgen_eval.execution.torch import context as torch_context
    from tabgen_eval.execution.torch.distance import ops as distance_ops

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_context, "find_spec", lambda name: None)
    (r, s, schema) = frames()
    context = context_for(PRDC(dtype=dtype), r, s, schema)
    assert context.inputs.device.type == "cpu"

    def missing(name):
        raise ModuleNotFoundError("No module named 'triton'", name="triton")

    monkeypatch.setattr(distance_ops.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="tabgen-eval\\[cuda\\]"):
        distance_ops.cuda_kernels()

    def broken(name):
        raise ModuleNotFoundError("broken installed engine", name="triton.broken")

    monkeypatch.setattr(distance_ops.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="broken installed engine") as exc:
        distance_ops.cuda_kernels()
    assert exc.value.name == "triton.broken"


def test_shared_cpu_context_records_per_metric_implementations(torch):
    (r, s, schema) = frames(n=260)
    report = evaluate(
        r,
        s,
        schema,
        metrics=["shape", "prdc"],
        options=make_options(device="cpu", dtype="float64", uncertainty="both", n_replicates=2),
    )
    assert {
        name: info["implementation"] for name, info in report.provenance["metric_execution"].items()
    } == {"shape": "aten_cpp", "prdc": "aten_cpp"}
    details = report.provenance["metric_execution"]
    assert details["shape"]["implementation"] == "aten_cpp"
    assert details["prdc"]["implementation"] == "aten_cpp"
    memory = evaluate_memorization(
        r, s, schema, options=make_options(device="cpu", dtype="float64")
    )
    assert memory.provenance["metric_execution"]["dcr"]["implementation"] == "aten_cpp"
    assert memory.provenance["metric_execution"]["dcr"]["implementation"] == "aten_cpp"
    assert memory.provenance["metric_execution"]["dcr"]["device"] == "cpu"


@pytest.mark.gpu
def test_gpu_workspace_peak_and_budget_independence(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    rng = np.random.default_rng(91)
    r = pd.DataFrame(rng.normal(size=(900, 20)))
    s = pd.DataFrame(rng.normal(size=(700, 20)))
    schema = Schema(numeric=list(r.columns))
    a = PRDC(device=device, dtype="float64", gpu_workspace_mb=1)
    context = context_for(a, r, s, schema)
    rd = np.arange(len(r))[None]
    sd = np.arange(len(s))[None]
    context.compute(a, rd, sd)
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = context.compute(a, np.repeat(rd, 4, axis=0), np.repeat(sd, 4, axis=0))
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    assert peak <= 1024**2
    b = PRDC(device=device, dtype="float64", gpu_workspace_mb=16)
    other = context_for(b, r, s, schema).compute(b, rd, sd)
    for name in result:
        np.testing.assert_array_equal(result[name], np.repeat(other[name], 4))
    metric = DCR(device=device, dtype="float64", gpu_workspace_mb=1)
    context = context_for(metric, r, s, schema)
    context.compute_memorization_group([metric], rd, sd)
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    outcomes = context.compute_memorization_group(
        [metric], np.repeat(rd, 4, axis=0), np.repeat(sd, 4, axis=0)
    )
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) - base <= 1024**2
    for outcome in outcomes:
        assert outcome["dcr"].values.keys() == outcomes[0]["dcr"].values.keys()
        np.testing.assert_array_equal(
            list(outcome["dcr"].values.values()), list(outcomes[0]["dcr"].values.values())
        )


@pytest.mark.gpu
def test_auto_cuda_selection_and_nondefault_stream(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    (r, s, schema) = frames(n=260)
    previous_device = torch.cuda.current_device()
    precision = torch.get_float32_matmul_precision()
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        options = make_options(device=device, uncertainty="none")
        report = evaluate(r, s, schema, metrics=["prdc"], options=options)
        memory = evaluate_memorization(r, s, schema, options=options)
    stream.synchronize()
    assert torch.cuda.current_device() == previous_device
    assert torch.get_float32_matmul_precision() == precision
    assert all(
        v["implementation"] == "triton" for v in report.provenance["metric_execution"].values()
    )
    assert memory.provenance["metric_execution"]["dcr"]["implementation"] == "triton"
    assert memory.provenance["metric_execution"]["dcr"]["device"] == device
    with torch.cuda.device(device):
        automatic = context_for(PRDC(), r, s, schema)
    assert str(automatic.inputs.device) == "cpu"
    assert automatic.execution_info_for(PRDC())["routing_reason"] == "small_workload_cpu"
