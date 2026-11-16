"""Descriptive Python fidelity tables share the existing numerical reductions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.python import conditional, fidelity
from tabgen_eval.execution.python.stats import safe_nanmean
from tabgen_eval.execution.runtime.details import DetailedOutcome
from tabgen_eval.preprocessing.categories import category_key

pytestmark = pytest.mark.unit


def _frames():
    labels = [1] * 10 + [1.0] * 6 + ["1"] * 3 + ["rare", None]
    other = [1] * 4 + [1.0] * 2 + ["1"] * 2 + ["candidate-only"] * 2 + [None]
    reference = pd.DataFrame(
        {
            "x": np.arange(21, dtype=float),
            "y": np.arange(21, dtype=float) ** 2,
            "c": pd.Series(labels, dtype=object),
        }
    )
    candidate = pd.DataFrame(
        {
            "x": np.linspace(-100, 100, 11),
            "y": np.arange(11, dtype=float) ** 2,
            "c": pd.Series(other, dtype=object),
        }
    )
    return reference, candidate, Schema(numeric=["x", "y"], categorical=["c"])


@pytest.mark.parametrize("name", ["shape", "trend", "conditional"])
def test_capture_preserves_scalar_values_exactly(name):
    reference, candidate, schema = _frames()
    function = (
        getattr(fidelity, "compute_" + name)
        if name != "conditional"
        else conditional.compute_conditional
    )
    options = {} if name != "conditional" else dict(num_bins=3, min_count=3, rare_threshold=0.2)
    original = function(reference, candidate, schema, **options)
    captured = function(reference, candidate, schema, details=True, **options)
    assert isinstance(captured, DetailedOutcome)
    assert original.keys() == captured.values.keys()
    for key, value in original.items():
        np.testing.assert_equal(captured.values[key], value)
    assert captured.details.metadata["column_names"] == ("x", "y", "c")


def test_shape_and_trend_tables_retain_all_contributing_scores_and_ids():
    reference, candidate, schema = _frames()
    shape = fidelity.compute_shape(reference, candidate, schema, details=True)
    columns = shape.details.tables["columns"].columns
    np.testing.assert_array_equal(columns["column_index"], [0, 1, 2])
    assert safe_nanmean(columns["score"]) == shape.values["shape"]
    trend = fidelity.compute_trend(reference, candidate, schema, details=True)
    pairs = trend.details.tables["pairs"].columns
    np.testing.assert_array_equal(pairs["column_1_index"], [0, 0, 1])
    np.testing.assert_array_equal(pairs["column_2_index"], [1, 2, 2])
    assert safe_nanmean(pairs["score"]) == trend.values["trend"]


def test_conditional_tables_reconstruct_both_levels_of_aggregation():
    reference, candidate, schema = _frames()
    result = conditional.compute_conditional(
        reference, candidate, schema, num_bins=3, min_count=3, rare_threshold=0.2, details=True
    )
    columns = result.details.tables["columns"].columns
    slices = result.details.tables["slices"].columns
    for column in columns["column_index"]:
        selected = (slices["column_index"] == column) & slices["eligible"]
        for field, prefix in [
            ("shape", "cond_shape"),
            ("trend", "cond_trend"),
            ("recovery", "recovery"),
        ]:
            for suffix, mask in [("all", selected), ("rare", selected & slices["rare"])]:
                np.testing.assert_equal(
                    columns[f"{prefix}_{suffix}"][column], safe_nanmean(slices[field][mask])
                )
    for name, value in result.values.items():
        np.testing.assert_equal(value, safe_nanmean(columns[name]))
    cat_rows = slices["column_index"] == 2
    assert np.all(slices["reference_total"][cat_rows] == 20)
    assert np.all(slices["candidate_total"][cat_rows] == 10)
    vocabulary = result.details.metadata["category_values"][2]
    represented = [vocabulary[code] for code in slices["label_code"][cat_rows]]
    assert {category_key(v) for v in represented} == {
        category_key(1),
        category_key(1.0),
        category_key("1"),
        category_key("rare"),
    }
    excluded = cat_rows & ~slices["eligible"]
    assert excluded.sum() == 1
    assert slices["reference_count"][excluded].item() == 1
    assert slices["rare"][excluded].item()
    for field in ["shape", "trend", "recovery", "lower", "upper"]:
        assert np.isnan(slices[field][excluded]).all()


def test_numeric_interval_metadata_reuses_actual_reference_edges():
    reference, candidate, schema = _frames()
    result = conditional.compute_conditional(
        reference, candidate, schema, num_bins=3, min_count=3, rare_threshold=0.2, details=True
    )
    slices = result.details.tables["slices"].columns
    rows = slices["column_index"] == 0
    np.testing.assert_allclose(slices["lower"][rows], [0, 20 / 3, 40 / 3])
    np.testing.assert_allclose(slices["upper"][rows], [20 / 3, 40 / 3, 20])
    np.testing.assert_array_equal(slices["left_closed"][rows], [True, False, False])
    assert slices["right_closed"][rows].all()
    assert slices["reference_count"][rows].sum() == len(reference)
    assert slices["candidate_count"][rows].sum() == len(candidate)


@pytest.mark.parametrize("width", [1, 2])
def test_detail_scores_mask_structurally_unavailable_conditional_statistics(width):
    reference = pd.DataFrame({"c": ["a"] * 4, "x": [0.0, 1.0, 2.0, 3.0]})
    candidate = pd.DataFrame({"c": ["other"] * 4, "x": [0.0, 1.0, 2.0, 3.0]})
    schema = Schema(numeric=["x"] if width == 2 else [], categorical=["c"])
    options = dict(num_bins=2, min_count=1, rare_threshold=0.2)
    original = conditional.compute_conditional(reference, candidate, schema, **options)
    detailed = conditional.compute_conditional(
        reference, candidate, schema, details=True, **options
    )
    for name, value in original.items():
        np.testing.assert_equal(value, detailed.values[name])
    columns = detailed.details.tables["columns"].columns
    slices = detailed.details.tables["slices"].columns
    for name in ["cond_trend_all", "cond_trend_rare"]:
        assert np.isnan(columns[name]).all()
    assert np.isnan(slices["trend"]).all()
    if width == 1:
        for name in original:
            if not name.startswith("recovery"):
                assert np.isnan(columns[name]).all()
        assert np.isnan(slices["shape"]).all()
        np.testing.assert_array_equal(slices["recovery"], [0.0])


def test_capture_does_not_repeat_existing_slice_metric_computations(monkeypatch):
    reference, candidate, schema = _frames()
    options = dict(num_bins=3, min_count=3, rare_threshold=0.2)
    counts = {"shape": 0, "trend": 0}
    for name in counts:
        original = getattr(conditional._Encoded, "slice_" + name)

        def counted(self, *args, _name=name, _original=original):
            counts[_name] += 1
            return _original(self, *args)

        monkeypatch.setattr(conditional._Encoded, "slice_" + name, counted)
    conditional.compute_conditional(reference, candidate, schema, **options)
    expected = counts.copy()
    counts.update(shape=0, trend=0)
    conditional.compute_conditional(reference, candidate, schema, details=True, **options)
    assert counts == expected


def test_undefined_conditioning_column_is_retained_with_nan_column_scores():
    reference, candidate, schema = _frames()
    reference["missing"] = candidate["missing"] = np.nan
    schema.numeric.append("missing")
    result = conditional.compute_conditional(
        reference, candidate, schema, num_bins=3, min_count=3, rare_threshold=0.2, details=True
    )
    table = result.details.tables["columns"].columns
    assert len(table["column_index"]) == 4
    for name in result.values:
        assert np.isnan(table[name][2])
    assert not (result.details.tables["slices"].columns["column_index"] == 2).any()
