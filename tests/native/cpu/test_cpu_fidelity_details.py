"""Same-pass descriptive CPU fidelity output and aggregate reduction contracts."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.execution.torch.cpu.executor import CppExecutor
from tabgen_eval.metrics import Conditional, Shape, Trend
from tabgen_eval.preprocessing.categories import MISSING_CATEGORY, factorize_categories

pytestmark = pytest.mark.unit


def _frames():
    rng = np.random.default_rng(7013)
    real = pd.DataFrame(rng.normal(size=(140, 3)), columns=["x", "y", "z"])
    generated = pd.DataFrame(rng.normal(size=(157, 3)), columns=["x", "y", "z"])
    for table in [real, generated]:
        table["a"] = rng.choice(["A", "B", "C"], len(table))
        table["b"] = rng.choice(["nan", "<MISSING>", " C "], len(table))
        table.loc[::9, "x"] = np.nan
        table.loc[::13, "y"] = np.nan
        table.loc[::7, "a"] = None
        table["z"] = 1.0
    return real, generated, Schema(numeric=["x", "y", "z"], categorical=["a", "b"])


@pytest.mark.parametrize("metric_class", [Shape, Trend, Conditional])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("ideal", [False, True])
def test_detailed_scalars_preserve_exact_bits_and_source_arrays(
    torch, metric_class, dtype, workers, ideal
):
    real, generated, schema = _frames()
    metric = metric_class(device="cpu", dtype=dtype)
    context = context_for(metric, real, generated, schema, n_jobs=workers)
    rd = np.r_[np.arange(127), [1, 1, 4]].astype(np.int64)
    sd = np.r_[np.arange(113), [2, 2, 5, 5, 5]].astype(np.int64)
    before = [
        tensor.clone()
        for table in [context.reference, context.candidate]
        for tensor in [table.numeric, table.categorical, table.complete]
    ]
    expected = context.compute(metric, rd[None, :], sd[None, :], ideal=ideal)
    outcome = CppExecutor(context).compute_detailed(metric, rd, sd, ideal=ideal)
    np.testing.assert_array_equal(
        np.asarray(list(outcome.values.values())).view(np.uint64),
        np.asarray([expected[name][0] for name in metric.stat_names]).view(np.uint64),
    )
    after = [
        tensor
        for table in [context.reference, context.candidate]
        for tensor in [table.numeric, table.categorical, table.complete]
    ]
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    if metric.name == "shape":
        columns = outcome.details.tables["columns"].columns
        np.testing.assert_array_equal(columns["column_index"], np.arange(5))
        assert len(columns["score"]) == 5
    elif metric.name == "trend":
        pairs = outcome.details.tables["pairs"].columns
        expected_pairs = np.array(list(combinations(range(5), 2)))
        np.testing.assert_array_equal(pairs["column_1_index"], expected_pairs[:, 0])
        np.testing.assert_array_equal(pairs["column_2_index"], expected_pairs[:, 1])
        assert len(pairs["score"]) == 10


def _conditional_frames():
    real = pd.DataFrame(
        {
            "x": np.arange(180, dtype=float) / 5,
            "a": ["A"] * 90 + ["B"] * 60 + ["C"] * 27 + ["tiny"] * 3,
            "b": ["one"] * 100 + ["two"] * 60 + ["three"] * 20,
        }
    )
    generated = pd.DataFrame(
        {
            "x": np.linspace(-10, 50, 180),
            "a": ["A"] * 120 + ["B"] * 45 + ["new"] * 5 + [None] * 10,
            "b": ["one"] * 50 + ["two"] * 50 + ["three"] * 80,
        }
    )
    return real, generated, Schema(numeric=["x"], categorical=["a", "b"])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_slice_counts_hierarchy_exclusions_and_draw_fitted_intervals(torch, dtype):
    real, generated, schema = _conditional_frames()
    metric = Conditional(device="cpu", dtype=dtype, min_count=10, rare_threshold=0.2)
    context = context_for(metric, real, generated, schema, n_jobs=4)
    outcome = CppExecutor(context).compute_detailed(metric, np.arange(180), np.arange(180))
    columns = outcome.details.tables["columns"].columns
    slices = outcome.details.tables["slices"].columns
    (codes, _), _ = factorize_categories(real["a"], generated["a"])
    a = slices["column_index"] == 1
    assert slices["reference_count"][a].tolist() == [90, 60, 27, 3]
    assert slices["candidate_count"][a].tolist() == [120, 45, 0, 0]
    assert (slices["reference_total"][a] == 180).all()
    assert (slices["candidate_total"][a] == 170).all()
    assert slices["candidate_count"][a].sum() == 165  # Candidate-only levels enter the total.
    tiny = a & (slices["label_code"] == codes[-1])
    assert not slices["eligible"][tiny].item()
    assert slices["rare"][tiny].item()
    for stat in ["shape", "trend", "recovery"]:
        assert np.isnan(slices[stat][tiny]).all()
    absent = a & (slices["label_code"] == codes[-4])
    assert slices["eligible"][absent].item()
    assert slices["candidate_count"][absent].item() == 0
    assert slices["shape"][absent].item() == 0
    assert slices["trend"][absent].item() == 0
    assert slices["recovery"][absent].item() == 0
    assert np.isnan(slices["lower"][a]).all()
    assert not slices["left_closed"][a].any()
    assert not slices["right_closed"][a].any()
    numeric = slices["column_index"] == 0
    assert slices["lower"][numeric][0] == 0
    assert slices["upper"][numeric][-1] == np.asarray(35.8, dtype=dtype)
    assert slices["left_closed"][numeric].sum() == 1
    assert slices["right_closed"][numeric].all()
    tolerance = 2e-6 if dtype == "float32" else 1e-14
    for stat, aggregate in [
        ("shape", "cond_shape"),
        ("trend", "cond_trend"),
        ("recovery", "recovery"),
    ]:
        for rare in [False, True]:
            key = f"{aggregate}_{'rare' if rare else 'all'}"
            for col in range(3):
                selected = (slices["column_index"] == col) & slices["eligible"]
                if rare:
                    selected &= slices["rare"]
                values = slices[stat][selected]
                finite = values[np.isfinite(values)]
                expected = float(finite.mean()) if len(finite) else np.nan
                np.testing.assert_allclose(
                    columns[key][col], expected, atol=tolerance, equal_nan=True
                )
            finite = columns[key][np.isfinite(columns[key])]
            expected = float(finite.mean()) if len(finite) else np.nan
            np.testing.assert_allclose(
                outcome.values[key], expected, atol=tolerance, equal_nan=True
            )
    flattened = slices["shape"][slices["eligible"]]
    assert abs(np.nanmean(flattened) - outcome.values["cond_shape_all"]) > 1e-3


def test_typed_category_ids_counts_and_strict_rarity_threshold(torch):
    labels = np.empty(100, dtype=object)
    labels[:] = (
        [1] * 10 + ["1"] * 20 + [1.0] * 20 + [True] * 20 + [MISSING_CATEGORY] * 20 + [None] * 10
    )
    real = pd.DataFrame({"x": np.arange(100, dtype=float), "c": labels})
    generated = real.iloc[::-1].reset_index(drop=True)
    schema = Schema(numeric=["x"], categorical=["c"])
    metric = Conditional(device="cpu", min_count=15, rare_threshold=2 / 9)
    context = context_for(metric, real, generated, schema)
    result = CppExecutor(context).compute_detailed(metric, np.arange(100), np.arange(100))
    table = result.details.tables["slices"].columns
    cat = table["column_index"] == 1
    np.testing.assert_array_equal(table["label_code"][cat], np.arange(5))
    np.testing.assert_array_equal(table["reference_count"][cat], [10, 20, 20, 20, 20])
    np.testing.assert_array_equal(table["candidate_count"][cat], [10, 20, 20, 20, 20])
    np.testing.assert_array_equal(table["eligible"][cat], [False, True, True, True, True])
    np.testing.assert_array_equal(table["rare"][cat], [True, False, False, False, False])
    assert table["reference_total"][cat].tolist() == [90] * 5
    repeated = CppExecutor(context).compute_detailed(
        metric,
        np.array([0, 0, 1, 10, 10, 30, 30, 30, 50, 70]),
        np.arange(10, 15),
    )
    counts = repeated.details.tables["slices"].columns
    selected = counts["column_index"] == 1
    np.testing.assert_array_equal(counts["reference_count"][selected], [3, 2, 3, 1, 1])
    np.testing.assert_array_equal(counts["candidate_count"][selected], [0, 0, 0, 0, 5])
    assert (counts["reference_total"][selected] == 10).all()
    assert (counts["candidate_total"][selected] == 5).all()


def test_individual_shape_and_trend_scores_have_analytic_values(torch):
    real = pd.DataFrame({"x": [0.0, 1, 2, 3], "y": [0.0, 1, 2, 3], "c": ["A", "A", "B", "B"]})
    generated = pd.DataFrame({"x": [0.0, 1, 2, 3], "y": [0.0, -1, -2, -3], "c": ["A"] * 4})
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    for cls, table, expected in [
        (Shape, "columns", [1, 0.25, 0.5]),
        (Trend, "pairs", [0, 0.5, 0.25]),
    ]:
        metric = cls(device="cpu")
        context = context_for(metric, real, generated, schema)
        result = CppExecutor(context).compute_detailed(metric, np.arange(4), np.arange(4))
        np.testing.assert_allclose(
            result.details.tables[table].columns["score"], expected, atol=1e-15
        )


def test_details_use_one_native_operator_and_validate_single_draw(torch, monkeypatch):
    real, generated, schema = _frames()
    metric = Shape(device="cpu")
    executor = CppExecutor(context_for(metric, real, generated, schema))
    operator = executor.ops.fidelity_details
    calls = []

    def detailed(*args):
        calls.append(1)
        return operator(*args)

    executor.ops = SimpleNamespace(fidelity_details=detailed)
    result = executor.compute_detailed(metric, np.arange(120), np.arange(130))
    assert len(calls) == 1
    assert result.values["shape"] > 0
    with pytest.raises(ValueError, match="one-dimensional"):
        executor.compute_detailed(metric, np.arange(2)[None, :], np.arange(2))
    with pytest.raises(ValueError, match="outside"):
        executor.compute_detailed(metric, np.array([-1]), np.arange(2))


@pytest.mark.parametrize("metric_class", [Shape, Trend, Conditional])
def test_empty_draws_return_typed_empty_or_undefined_details(torch, metric_class):
    real, generated, schema = _frames()
    metric = metric_class(device="cpu")
    executor = CppExecutor(context_for(metric, real, generated, schema))
    result = executor.compute_detailed(
        metric, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    assert all(np.isnan(v) for v in result.values.values())
    if metric.name == "conditional_fidelity":
        slices = result.details.tables["slices"].columns
        assert all(len(v) == 0 for v in slices.values())
        assert slices["eligible"].dtype == bool


@pytest.mark.parametrize("width", [1, 2])
def test_structural_detail_masks_preserve_raw_scalar_reductions(torch, width):
    real = pd.DataFrame({"c": ["A"] * 10 + ["B"] * 10})
    generated = pd.DataFrame({"c": ["A"] * 20})
    if width == 2:
        real["d"] = np.tile(["x", "y"], 10)
        generated["d"] = np.tile(["x", "y"], 10)
    schema = Schema(categorical=list(real.columns))
    metric = Conditional(device="cpu", min_count=1)
    context = context_for(metric, real, generated, schema)
    expected = context.compute(metric, np.arange(20)[None, :], np.arange(20)[None, :])
    detailed = CppExecutor(context).compute_detailed(metric, np.arange(20), np.arange(20))
    np.testing.assert_array_equal(
        np.asarray(list(detailed.values.values())).view(np.uint64),
        np.asarray([expected[name][0] for name in metric.stat_names]).view(np.uint64),
    )
    columns = detailed.details.tables["columns"].columns
    slices = detailed.details.tables["slices"].columns
    assert np.isnan(columns["cond_trend_all"]).all()
    assert np.isnan(columns["cond_trend_rare"]).all()
    assert np.isnan(slices["trend"]).all()
    if width == 1:
        assert all(
            np.isnan(columns[name]).all()
            for name in metric.stat_names
            if not name.startswith("recovery")
        )
        assert np.isnan(slices["shape"]).all()
        np.testing.assert_array_equal(slices["recovery"], [2.0, 0.0])
        np.testing.assert_array_equal(columns["recovery_all"], [1.0])
    assert slices["eligible"].all()
    assert (slices["reference_count"] > 0).all()
