"""Named fidelity table views and typed, additive detail serialization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.python.conditional import compute_conditional
from tabgen_eval.execution.python.fidelity import compute_shape, compute_trend
from tabgen_eval.execution.runtime.details import FidelityDetails
from tabgen_eval.metrics import Direction
from tabgen_eval.preprocessing.categories import MISSING_CATEGORY, category_key
from tabgen_eval.report import MetricResult, Report, StatResult

pytestmark = pytest.mark.unit


def _report(metric, details):
    return Report([MetricResult(metric, "fidelity", details=details)], {}, {})


def _typed_conditional():
    labels = [1, 1.0, "1", True, "<MISSING>", MISSING_CATEGORY]
    values = np.empty(60, dtype=object)
    for i in range(len(values)):
        values[i] = labels[i % len(labels)]
    frame = pd.DataFrame({"x": np.arange(60, dtype=float), "y": np.arange(60, dtype=float) ** 2})
    frame["c"] = pd.Series(values, dtype=object)
    result = compute_conditional(
        frame,
        frame,
        Schema(numeric=["x", "y"], categorical=["c"]),
        num_bins=3,
        min_count=2,
        rare_threshold=0.2,
        details=True,
    )
    return _report("conditional_fidelity", result.details), labels


def test_slice_view_preserves_typed_labels_and_actual_numeric_intervals():
    report, labels = _typed_conditional()
    table = report.details("conditional_fidelity", level="slice")
    observed = table.loc[table["column"] == "c", "label"]
    assert observed.dtype == object
    assert {category_key(value) for value in observed} == {category_key(value) for value in labels}
    assert any(type(value) is int for value in observed)
    assert any(type(value) is float for value in observed)
    assert any(type(value) is bool for value in observed)
    assert "column_index" not in table and "label_code" not in table
    intervals = table.loc[table["column"] == "x", "label"].tolist()
    assert all(isinstance(value, pd.Interval) for value in intervals)
    assert intervals[0].closed == "both"
    assert intervals[1].closed == intervals[2].closed == "right"
    assert intervals[0].left == 0.0
    assert intervals[-1].right == 59.0
    assert table.attrs["metric"] == "conditional_fidelity"
    assert table.attrs["level"] == "slice"


def test_default_views_are_named_columns_and_pairs_without_derived_summaries():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 1.0]})
    schema = Schema(numeric=["x", "y"])
    shape = _report("shape", compute_shape(frame, frame, schema, details=True).details)
    assert shape.details("shape").columns.tolist() == ["column", "score"]
    assert shape.details("shape")["column"].tolist() == ["x", "y"]
    trend = _report("trend", compute_trend(frame, frame, schema, details=True).details)
    pairs = trend.details("trend")
    assert pairs.columns.tolist() == ["column_1", "column_2", "score"]
    assert pairs.loc[0, "column_1"] == "x"
    assert pairs.loc[0, "column_2"] == "y"
    with pytest.raises(ValueError, match="available"):
        trend.details("trend", level="column")
    with pytest.raises(ValueError, match="detail level"):
        trend.details("trend", level="everything")


def test_table_views_are_copies_of_scores_and_metadata():
    report, _ = _typed_conditional()
    first = report.details("conditional_fidelity")
    first.iloc[0, 1] = -1234.0
    first.attrs["metadata"]["column_names"] = ("changed",)
    second = report.details("conditional_fidelity")
    assert second.iloc[0, 1] != -1234.0
    assert second.attrs["metadata"]["column_names"] == ("x", "y", "c")


def test_json_retains_typed_vocabularies_and_serializes_nonfinite_detail_values_as_null():
    report, _ = _typed_conditional()
    encoded = json.loads(report.to_json())
    detail = encoded["details"]["conditional_fidelity"]
    vocabulary = detail["metadata"]["category_values"]["2"]
    assert [entry["family"] for entry in vocabulary] == [
        "integer",
        "float",
        "string",
        "boolean",
        "string",
        "missing-category",
    ]
    assert vocabulary[0]["display"] == 1
    assert vocabulary[1]["display"] == 1.0
    assert vocabulary[2]["display"] == "1"
    assert vocabulary[4]["display"] == "<MISSING>"
    assert vocabulary[5]["display"] == "<missing category>"
    slices = detail["tables"]["slices"]["columns"]
    for i, column in enumerate(slices["column_index"]):
        if column == 2:
            assert slices["lower"][i] is None
            assert slices["upper"][i] is None
    # Strict encoding rejects nonstandard NaN/Infinity tokens anywhere in details.
    json.dumps(encoded["details"], allow_nan=False)


def test_details_are_additive_to_existing_scalar_json_and_absent_without_capture():
    statistic = StatResult(
        name="shape",
        display="Shape",
        value=0.75,
        ci_low=np.nan,
        ci_high=np.nan,
        bootstrap_median=np.nan,
        ideal=np.nan,
        ideal_ci_low=np.nan,
        ideal_ci_high=np.nan,
        ideal_bootstrap_median=np.nan,
        direction=Direction.HIGHER_BETTER,
        target=None,
        n_holdout=200,
        n_generated=200,
        n_eff=100,
    )
    metric = MetricResult("shape", "fidelity", stats={"shape": statistic})
    report = Report([metric], {}, {})
    before = json.loads(report.to_json())
    assert "details" not in before
    metric.details = FidelityDetails(metadata={"column_names": ()})
    after = json.loads(report.to_json())
    assert after["results"] == before["results"]
    assert after["details"]["shape"]["tables"] == {}


def test_missing_capture_and_unknown_metrics_have_actionable_errors():
    report = _report("shape", None)
    with pytest.raises(ValueError, match="details=True"):
        report.details("shape")
    with pytest.raises(KeyError, match="metric"):
        report.details("unknown")
