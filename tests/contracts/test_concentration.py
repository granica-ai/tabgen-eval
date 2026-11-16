"""Local mass, geometry, radius selection and shared-profile execution contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.python.concentration import (
    concentration,
    local_counts,
    outcome,
    pair_indices,
    paired_distances,
    summarize_counts,
)
from tabgen_eval.metrics.base import Direction
from tabgen_eval.metrics.concentration import Concentration
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def oracle(numeric, categorical, radii):
    n = len(numeric)
    distances = np.zeros((n, n), dtype=numeric.dtype)
    for j in range(numeric.shape[1]):
        delta = numeric[:, j, None] - numeric[None, :, j]
        distances += delta * delta
    categories = np.zeros((n, n), dtype=np.int64)
    for j in range(categorical.shape[1]):
        categories += 2 * (categorical[:, j, None] != categorical[None, :, j])
    distances += categories.astype(numeric.dtype)
    np.fill_diagonal(distances, np.inf)
    thresholds = np.square(np.asarray(radii, dtype=np.float64)).astype(numeric.dtype)
    return np.asarray([(distances <= threshold).sum(axis=1) for threshold in thresholds])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_counts_match_direct_oracle_and_monotone_radii(dtype):
    rng = np.random.default_rng(12)
    numbers = rng.normal(size=(79, 3)).astype(dtype)
    categories = rng.integers(0, 4, size=(79, 2))
    numbers[1], categories[1] = numbers[0], categories[0]
    radii = [0.0, 1e-3, 1.0, 2.0, 100.0]
    result = local_counts(numbers, categories, radii, tile_size=13)
    np.testing.assert_array_equal(result, oracle(numbers, categories, radii))
    assert np.all(np.diff(result, axis=0) >= 0)
    assert result[0, 0] >= 1  # An identical other record is not positional self.
    assert np.all(result[-1] == 78)


def test_two_clusters_have_probability_weighted_mass():
    # Four rows in one cluster and six in another: each center sees 3 or 5 others.
    numbers = np.asarray([[0.0]] * 4 + [[1.0]] * 6)
    counts = local_counts(numbers, np.empty((10, 0), dtype=np.int64), [1e-3])
    np.testing.assert_array_equal(counts[0], [3] * 4 + [5] * 6)
    summary = summarize_counts(counts, 0.95)
    assert summary[0, 0] == pytest.approx((4 * 3 + 6 * 5) / 90)
    assert summary[0, 1] == pytest.approx(5 / 9)


@pytest.mark.parametrize("n", [0, 1])
def test_insufficient_records_are_undefined(n):
    frame = pd.DataFrame({"x": np.arange(n, dtype=float)})
    result = concentration(frame, frame, Schema(numeric=["x"]), Concentration(), details=True)
    assert all(np.isnan(value) for value in result.values.values())
    assert np.all(result.details.tables["radii"].columns["candidate_n"] == n)


def test_generated_only_categories_keep_distinct_typed_identities():
    reference = pd.DataFrame({"c": pd.Series(["real"] * 6, dtype=object)})
    generated = pd.DataFrame({"c": pd.Series([1, "1", 1.0, True, " A ", "A"], dtype=object)})
    metric = Concentration(radius_quantiles=())
    result = concentration(reference, generated, Schema(categorical=["c"]), metric, details=True)
    assert result.values["mean_radii_1e-3"] == pytest.approx(1e-8 / (1 + 1e-8))
    assert set(result.values) == {"mean_radii_1e-3"}
    table = result.details.tables["radii"].columns
    assert table["reference_mean"][0] == 1
    assert table["candidate_mean"][0] == 0


def test_collapsed_generated_data_has_large_finite_ratio_and_matching_data_one():
    reference = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    generated = pd.DataFrame({"x": np.full(100, 0.5)})
    schema = Schema(numeric=["x"])
    metric = Concentration(radius_quantiles=())
    assert concentration(reference, generated, schema, metric)["mean_radii_1e-3"] == pytest.approx(
        (1 + 1e-8) / 1e-8
    )
    assert concentration(reference, reference, schema, metric)["mean_radii_1e-3"] == 1
    assert all(spec.direction is Direction.TARGET and spec.target == 1 for spec in metric.stats)


def test_pair_sample_is_reproducible_distinct_and_leaves_global_rng_untouched():
    np.random.seed(987)
    before = np.random.get_state()
    left, right = pair_indices(25, 4096, 42)
    after = np.random.get_state()
    assert np.all(left != right)
    np.testing.assert_array_equal(left, pair_indices(25, 4096, 42)[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_radii_and_numeric_scale_depend_only_on_reference():
    reference = pd.DataFrame({"x": np.arange(30, dtype=float)})
    metric = Concentration(radius_pairs=1024)
    schema = Schema(numeric=["x"])
    a = concentration(reference, reference, schema, metric, details=True)
    b = concentration(reference, pd.DataFrame({"x": [1e8] * 30}), schema, metric, details=True)
    ca, cb = a.details.tables["radii"].columns, b.details.tables["radii"].columns
    np.testing.assert_array_equal(ca["radius"], cb["radius"])
    np.testing.assert_array_equal(ca["reference_mean"], cb["reference_mean"])
    assert list(ca["radius_id"]) == ["radii_1e-3", "q05", "q25"]
    assert b.values["mean_radii_1e-3"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixed_radii": [-1]},
        {"fixed_radii": [np.inf]},
        {"fixed_radii": [], "radius_quantiles": []},
        {"radius_quantiles": [1.1]},
        {"radius_quantiles": [0.1, 0.1]},
        {"upper_quantile": np.nan},
        {"radius_pairs": 0},
        {"radius_pairs": True},
        {"radius_seed": -1},
    ],
)
def test_invalid_metric_options_rejected(kwargs):
    with pytest.raises(ValueError):
        Concentration(**kwargs)


def make_executor(reference, candidate, schema, device, dtype):
    from tabgen_eval.execution.torch.concentration import ConcentrationExecutor
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.preprocessing.columns import PreparedColumns

    columns = PreparedColumns.prepare(reference, candidate, schema)
    context = TorchContext.from_columns(columns, device=device, dtype=dtype, n_jobs=2)
    return ConcentrationExecutor(context)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_executor_matches_python_and_reuses_reference_profile(device, dtype, monkeypatch):
    rng = np.random.default_rng(99)
    reference = pd.DataFrame({"x": rng.normal(size=89), "c": rng.choice(["a", "b"], size=89)})
    candidate = pd.DataFrame({"x": rng.normal(size=41), "c": rng.choice(["new1", "new2"], size=41)})
    schema = Schema(numeric=["x"], categorical=["c"])
    metric = Concentration(dtype=dtype)
    executor = make_executor(reference, candidate, schema, device, dtype)
    count_calls = []
    original = executor._counts

    def counted(*args):
        count_calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(executor, "_counts", counted)
    rd, sd = np.arange(43), np.arange(41)
    result = executor.compute_detailed(metric, rd, sd)
    expected = concentration(reference.iloc[rd], candidate, schema, metric, details=True)
    for name in metric.stat_names:
        assert result.values[name] == pytest.approx(expected.values[name], abs=1e-12)
    executor.compute(metric, rd[None], np.arange(43, 86)[None], ideal=True)
    assert count_calls == [43, 41, 43]
    # Next reference replaces the cache rather than accumulating one profile per draw.
    executor.compute(metric, np.arange(1, 44)[None], sd[None])
    assert count_calls[-2:] == [43, 41]


def test_workspace_is_checked_before_source_gather(monkeypatch):
    import torch

    frame = pd.DataFrame({"x": np.arange(10, dtype=float)})
    metric = Concentration(radius_pairs=1_000_000)
    executor = make_executor(frame, frame, Schema(numeric=["x"]), "cpu", "float64")
    executor.context.gpu_workspace_mb = 1
    executor._reserve(metric, 10, 10)  # CPU does not use the CUDA workspace cap.
    executor.context.inputs = SimpleNamespace(device=torch.device("cuda"))

    def unexpected(*args):
        raise AssertionError("Workspace rejection must precede gather")

    monkeypatch.setattr(executor, "_sample", unexpected)
    with pytest.raises(MemoryError, match="workspace"):
        executor.compute(metric, np.arange(10)[None], np.arange(10)[None])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_executor_preserves_sources_and_filters_missing_rows(device, dtype):
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame({"x": [0.0, 0.0, 1.0, np.nan], "c": ["a", "a", "b", "a"]})
    generated = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0], "c": ["new", "new", "other", None]})
    metric = Concentration(dtype=dtype, radius_quantiles=())
    executor = make_executor(
        frame, generated, Schema(numeric=["x"], categorical=["c"]), device, dtype
    )
    source_numeric = executor.context.reference.numeric.clone()
    source_codes = executor.context.candidate.categorical.clone()
    result = executor.compute_detailed(metric, np.arange(4), np.arange(4))
    table = result.details.tables["radii"].columns
    assert table["reference_n"][0] == table["candidate_n"][0] == 3
    assert table["candidate_mean"][0] == pytest.approx(1 / 3)
    torch.testing.assert_close(executor.context.reference.numeric, source_numeric, equal_nan=True)
    torch.testing.assert_close(executor.context.candidate.categorical, source_codes)


def test_float32_category_rounding_matches_radius_selection_at_boundary(device):
    import torch

    numeric = np.asarray([[0], [np.sqrt(np.float32(0.95046365))]], dtype=np.float32)
    categories = np.asarray([[0] * 5, [1] * 5], dtype=np.int64)
    expected = np.float32(numeric[1, 0] ** 2 + np.float32(10))
    radius = np.sqrt(float(expected))
    left, right = np.asarray([0]), np.asarray([1])
    np.testing.assert_array_equal(paired_distances(numeric, categories, left, right), [expected])
    np.testing.assert_array_equal(local_counts(numeric, categories, [radius]), [[1, 1]])
    a, c = torch.as_tensor(numeric, device=device), torch.as_tensor(categories, device=device)
    left_tensor, right_tensor = (
        torch.as_tensor(left, device=device),
        torch.as_tensor(right, device=device),
    )
    threshold = torch.as_tensor([expected], device=device)
    if device == "cuda":
        from tabgen_eval.execution.torch.cuda.triton.concentration import (
            local_counts as native_counts,
        )
        from tabgen_eval.execution.torch.cuda.triton.concentration import (
            paired_distances as native_pairs,
        )

        actual_pair = native_pairs(a, c, left_tensor, right_tensor)
        actual_counts = native_counts(a, c, threshold)
    else:
        from tabgen_eval.execution.torch.cpu.extension import library

        actual_pair = library().paired_concentration_distances(a, c, left_tensor, right_tensor, 2)
        actual_counts = library().local_concentration_counts(a, c, threshold, 2)
    np.testing.assert_array_equal(actual_pair.cpu().numpy(), [expected])
    np.testing.assert_array_equal(actual_counts.cpu().numpy(), [[1, 1]])


@pytest.mark.parametrize("candidate", [[0.5, np.nan, np.nan], [np.nan] * 4])
def test_unusable_candidate_skips_quadratic_work(device, candidate, monkeypatch):
    reference = pd.DataFrame({"x": np.arange(100, dtype=float)})
    executor = make_executor(
        reference, pd.DataFrame({"x": candidate}), Schema(numeric=["x"]), device, "float64"
    )

    def unexpected(*args):
        raise AssertionError("An undefined comparison must not count reference pairs")

    monkeypatch.setattr(executor, "_counts", unexpected)
    result = executor.compute_detailed(Concentration(), np.arange(100), np.arange(len(candidate)))
    assert all(np.isnan(value) for value in result.values.values())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_multi_radius_counts_and_sampled_distances(dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tabgen_eval.execution.torch.cuda.triton.concentration import local_counts as gpu_counts
    from tabgen_eval.execution.torch.cuda.triton.concentration import paired_distances as gpu_pairs

    rng = np.random.default_rng(37)
    numeric = rng.normal(size=(147, 5)).astype(dtype)
    categories = rng.integers(0, 3, size=(147, 4), dtype=np.int64)
    numeric[1], categories[1] = numeric[0], categories[0]
    radii = np.asarray([0, 1e-3, 1, 2, 3, 20])
    a = torch.as_tensor(numeric, device="cuda")
    c = torch.as_tensor(categories, device="cuda")
    thresholds = torch.as_tensor((radii * radii).astype(dtype), device="cuda")
    actual = gpu_counts(a, c, thresholds).cpu().numpy()
    np.testing.assert_array_equal(actual, oracle(numeric, categories, radii))
    left, right = pair_indices(147, 1073, 23)
    actual_pairs = (
        gpu_pairs(a, c, torch.as_tensor(left, device="cuda"), torch.as_tensor(right, device="cuda"))
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(actual_pairs, paired_distances(numeric, categories, left, right))


def public_frames():
    rng = np.random.default_rng(258)
    holdout = pd.DataFrame({"x": rng.normal(size=80), "c": rng.choice(["a", "b"], size=80)})
    generated = pd.DataFrame({"x": rng.normal(size=37), "c": rng.choice(["a", "new"], size=37)})
    return holdout, generated, Schema(numeric=["x"], categorical=["c"])


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_public_python_native_parity_all_modes_and_compact_details(mode, device):
    holdout, generated, schema = public_frames()
    reports = []
    for placement in ("cpu", device):
        reports.append(
            evaluate(
                holdout,
                generated,
                schema=schema,
                metrics=["concentration"],
                options=make_options(
                    device=placement,
                    uncertainty=mode,
                    n_replicates=3,
                    seed=51,
                ),
            )
        )
    a, b = reports
    assert [result.name for result in a.results] == ["concentration"]
    for name in Concentration().stat_names:
        sa, sb = a[f"concentration/{name}"], b[f"concentration/{name}"]
        assert sa.value == pytest.approx(sb.value)
        assert sa.ideal == pytest.approx(sb.ideal)
        if mode != "none":
            np.testing.assert_allclose(
                sa.reference_distribution.samples, sb.reference_distribution.samples, atol=1e-15
            )
            assert not hasattr(sa, "comparison")
        if mode == "both":
            np.testing.assert_allclose(
                sa.generated_distribution.samples, sb.generated_distribution.samples, atol=1e-15
            )
    for report in reports:
        assert set(report.results[0].stats) == {
            "mean_radii_1e-3",
            "mean_q05",
            "upper_q05",
            "mean_q25",
            "upper_q25",
        }
        exported_stats = json.loads(report.to_json(include_replicates=True))
        assert "upper_radii_1e-3" not in json.dumps(exported_stats)
        table = report.details("concentration")
        assert len(table) == 3
        assert table.attrs["level"] == "radius"
        assert set(table.radius_id) == {"radii_1e-3", "q05", "q25"}
        n_res = 18 if mode == "both" else 37
        assert np.all(table.reference_n == n_res)
        assert np.all(table.candidate_n == n_res)
        assert all(stat.n_eff == n_res for _, stat in report.iter_stats())
        exported = json.loads(report.to_json(include_replicates=True))
        assert "radii" in exported["details"]["concentration"]["tables"]


def test_custom_metric_radii_survive_evaluation_and_calibration():
    holdout, generated, schema = public_frames()
    options = make_options(
        device="cpu",
        seed=19,
    )
    chosen = [
        configured_metric(
            Concentration, options, fixed_radii=[0, 0.1], radius_quantiles=[0.1], upper_quantile=0.9
        )
    ]
    report = evaluate(holdout, generated, schema=schema, metrics=chosen, options=options)
    expected = {
        "mean_radii_0",
        "upper_radii_0",
        "mean_radii_1e-1",
        "upper_radii_1e-1",
        "mean_q10",
        "upper_q10",
    }
    assert set(report.results[0].stats) == expected
    assert report.details("concentration").attrs["metadata"]["upper_quantile"] == 0.9
    assert report.details("concentration").attrs["metadata"]["epsilon"] == 1e-8
    reference = calibrate(holdout, schema=schema, metrics=chosen, options=options, n_replicates=3)
    assert set(summaries(reference.metrics["concentration"])) == expected
    assert reference.metadata["n_res"] == 40
    for distribution in summaries(reference.metrics["concentration"]).values():
        assert distribution.finite_count == 3


@pytest.mark.parametrize("large_generated_pool", [False, True])
def test_public_mode_resolution_and_reference_reuse_across_replicates(
    monkeypatch, large_generated_pool
):
    from tabgen_eval.execution.torch.concentration import ConcentrationExecutor

    holdout, generated, schema = public_frames()
    if large_generated_pool:
        generated = pd.concat([generated, generated], ignore_index=True)
    calls = []
    original = ConcentrationExecutor._counts

    def counted(self, *args):
        calls.append(len(args[0]))
        return original(self, *args)

    monkeypatch.setattr(ConcentrationExecutor, "_counts", counted)
    point = evaluate(
        holdout,
        generated,
        schema=schema,
        metrics=["concentration"],
        options=make_options(
            device="cpu",
        ),
    )
    assert len(calls) == 3
    calls.clear()
    reference = evaluate(
        holdout,
        generated,
        schema=schema,
        metrics=["concentration"],
        options=make_options(device="cpu", uncertainty="reference", n_replicates=4),
    )
    assert len(calls) == 2 + 2 * 4
    calls.clear()
    repeated = evaluate(
        holdout,
        generated,
        schema=schema,
        metrics=["concentration"],
        options=make_options(device="cpu", uncertainty="both", n_replicates=4),
    )
    assert len(calls) == 3 * 4
    n_res = 40 if large_generated_pool else 18
    assert calls == [n_res] * 12
    from tabgen_eval.execution.runtime.sampling import fixed_draw, paired_draw_batches

    draw = fixed_draw(len(holdout), len(generated), seed=42, uncertainty="both")
    values, ideals = [], []
    metric = Concentration()
    for batch in paired_draw_batches(
        draw, n_replicates=4, seed=42, batch_size=1, include_generated=True
    ):
        real = holdout.iloc[batch.reference[0]]
        values.append(
            metric.compute(
                real, generated.iloc[batch.generated[0]], schema, rng=np.random.default_rng(0)
            )
        )
        ideals.append(
            metric.compute(
                real, holdout.iloc[batch.real_candidate[0]], schema, rng=np.random.default_rng(0)
            )
        )
    for name in Concentration().stat_names:
        key = f"concentration/{name}"
        assert reference[key].value == point[key].value
        assert reference[key].ideal == reference[key].reference_distribution.mean
        assert repeated[key].value == pytest.approx(np.mean([row[name] for row in values]))
        assert repeated[key].ideal == pytest.approx(np.mean([row[name] for row in ideals]))


@pytest.mark.gpu
def test_public_cuda_real_only_calibration():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    holdout, _, schema = public_frames()
    result = calibrate(
        holdout,
        schema=schema,
        metrics=["concentration"],
        n_replicates=3,
        options=make_options(device="cuda"),
    )
    for distribution in summaries(result.metrics["concentration"]).values():
        assert distribution.finite_count == 3


def test_fixed_radius_names_follow_values_without_rounding_collisions():
    radii = (0.001, np.nextafter(0.001, 1.0), 0.0)
    metric = Concentration(fixed_radii=radii, radius_quantiles=())
    assert len(set(metric.stat_names)) == 5
    reference = pd.DataFrame({"x": [0.0, 0.001, 0.01, 1.0]})
    candidate = pd.DataFrame({"x": [0.0, 0.0, 0.5, 1.0]})
    schema = Schema(numeric=["x"])
    original = concentration(reference, candidate, schema, metric, details=True)
    ids = original.details.tables["radii"].columns["radius_id"]
    assert [float(label.removeprefix("radii_")) for label in ids] == list(radii)
    reordered = concentration(
        reference,
        candidate,
        schema,
        Concentration(fixed_radii=tuple(reversed(radii)), radius_quantiles=()),
    )
    assert original.values == reordered


def test_smoothed_ratio_details_and_fixed_epsilon():
    metric = Concentration(fixed_radii=(0.1,), radius_quantiles=())
    reference = np.array([[0.25, 0.5]])
    candidate = np.array([[0.5, 0.25]])
    result = outcome(metric, [0.1], reference, candidate, (10, 10), details=True)
    assert result.values == pytest.approx(
        {
            "mean_radii_1e-1": (0.5 + 1e-8) / (0.25 + 1e-8),
            "upper_radii_1e-1": (0.25 + 1e-8) / (0.5 + 1e-8),
        }
    )
    table = result.details.tables["radii"]
    assert table.columns["mean_ratio"][0] == result.values["mean_radii_1e-1"]
    assert table.columns["upper_ratio"][0] == result.values["upper_radii_1e-1"]
    assert "mean_gap" not in table.columns and "upper_gap" not in table.columns
    assert table.metadata["epsilon"] == 1e-8
    assert result.details.metadata["epsilon"] == 1e-8
    for mass in (0.0, 0.5, 1.0):
        profile = np.full((1, 2), mass)
        assert set(outcome(metric, [0.1], profile, profile, (10, 10)).values()) == {1.0}
    assert all(spec.bounds == (0.0, float("inf")) for spec in metric.stats)


@pytest.mark.parametrize("epsilon", [1e-8, 0.01])
def test_epsilon_is_not_configurable(epsilon):
    with pytest.raises(TypeError, match="unexpected kwarg 'epsilon'"):
        Concentration(epsilon=epsilon)
