"""Independent raw-data diagnostics use actual complete training support."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.diagnostics import DiagnosticReport, diagnose
from tabgen_eval.schema import Schema

pytestmark = pytest.mark.unit


def test_profiles_describe_inputs_without_cleaning_or_size_preferences():
    train = pd.DataFrame({"x": [0.0, 2.0, np.nan], "label": ["a", "b", "c"]})
    generated = pd.DataFrame({"x": [1.0, 1.0, np.nan, 3.0], "label": ["a", "a", "d", None]})
    original_train, original_generated = train.copy(deep=True), generated.copy(deep=True)
    report = diagnose(train, generated, schema=Schema(numeric=["x"], categorical=["label"]))
    assert isinstance(report, DiagnosticReport)
    assert report.inputs["train"]["n_rows"] == 3
    assert report.inputs["generated"]["n_rows"] == 4
    assert report.inputs["generated"]["n_missing_cells"] == 2
    assert report.inputs["generated"]["missing_cell_rate"] == 0.25
    assert report.inputs["generated"]["missing_row_rate"] == 0.5
    assert report.inputs["generated"]["n_duplicate_rows"] == 1
    assert report.inputs["generated"]["duplicate_row_rate"] == 0.25
    assert "size_ratio" not in report.to_json()
    assert not hasattr(report, "overall_score")
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(generated, original_generated)


def test_category_support_uses_every_training_row_and_excludes_missing_cells():
    train = pd.DataFrame({"category": [f"level-{i}" for i in range(1_000)]})
    generated = pd.DataFrame({"category": ["level-999", "outside", "outside", None]})
    report = diagnose(train, generated)
    support = report.categorical_support["category"]
    assert support["n_train_categories"] == 1_000
    assert support["n_unseen_categories"] == 1
    assert support["n_unseen_cells"] == 2
    assert support["unseen_cell_rate"] == pytest.approx(2 / 3)
    assert support["unseen_categories"] == [{"type": "string", "value": "outside"}]


def test_typed_category_identity_is_shared_by_support_and_duplicate_checks():
    train = pd.DataFrame({"category": pd.Series([True, 1, 1.0, "1"], dtype=object)})
    generated = pd.DataFrame({"category": pd.Series([True, 1, 1.0, "1", 2], dtype=object)})
    report = diagnose(train, generated)
    support = report.categorical_support["category"]
    assert support["n_train_categories"] == 4
    assert support["n_unseen_categories"] == 1
    assert support["unseen_categories"] == [{"type": "integer", "value": 2}]
    assert report.inputs["train"]["n_duplicate_rows"] == 0
    assert report.inputs["generated"]["n_duplicate_rows"] == 0


def test_structure_and_schema_differences_are_reported_instead_of_rejected():
    train = pd.DataFrame({"x": [1.0], "category": ["a"], "old": [1]})
    generated = pd.DataFrame({"extra": [5], "category": ["a"], "x": ["invalid"]})
    report = diagnose(
        train, generated, schema=Schema(numeric=["x"], categorical=["category", "old"])
    )
    assert report.structure["missing_columns"] == ["old"]
    assert report.structure["unexpected_columns"] == ["extra"]
    assert report.structure["column_order_matches"] is False
    assert report.structure["dtype_mismatches"]["x"] == {
        "train": str(train["x"].dtype),
        "generated": str(generated["x"].dtype),
    }
    assert report.inputs["generated"]["missing_schema_columns"] == ["old"]
    assert report.inputs["generated"]["undeclared_columns"] == ["extra"]
    assert report.numeric_support["x"]["n_generated_non_numeric"] == 1
    assert report.numeric_support["x"]["generated_min"] is None
    assert "old" not in report.categorical_support


def test_numeric_support_reports_observations_in_original_units():
    train = pd.DataFrame({"x": [0.0, 10.0, np.inf]})
    generated = pd.DataFrame({"x": [-1.0, 4.0, 11.0, -np.inf]})
    report = diagnose(train, generated, schema=Schema(numeric=["x"]))
    assert report.numeric_support["x"] == {
        "train_min": 0.0,
        "train_max": 10.0,
        "generated_min": -1.0,
        "generated_max": 11.0,
        "n_below_train_min": 1,
        "n_above_train_max": 1,
        "n_train_non_numeric": 0,
        "n_generated_non_numeric": 0,
        "n_train_nonfinite": 1,
        "n_generated_nonfinite": 1,
    }


def test_empty_inputs_have_unavailable_rates_and_strict_json():
    frame = pd.DataFrame({"x": pd.Series(dtype=float), "category": pd.Series(dtype=object)})
    report = diagnose(frame, frame, schema=Schema(numeric=["x"], categorical=["category"]))
    assert report.inputs["train"]["missing_cell_rate"] is None
    assert report.inputs["train"]["duplicate_row_rate"] is None
    assert report.categorical_support["category"]["unseen_cell_rate"] is None
    assert report.numeric_support["x"]["n_below_train_min"] is None
    assert json.loads(report.to_json())["report_type"] == "data_diagnostics"


def test_duplicate_column_names_are_observed_and_ambiguous_comparisons_omitted():
    train = pd.DataFrame([["a", "b"]], columns=["category", "category"])
    generated = pd.DataFrame({"category": ["a"]})
    report = diagnose(train, generated)
    assert report.structure["duplicate_train_columns"] == ["category"]
    assert report.categorical_support == {}
    assert len(report.inputs["train"]["columns"]) == 2


def test_reports_serialize_export_tables_and_do_not_expose_internal_data(tmp_path):
    train = pd.DataFrame({"category": ["a", "b"]})
    report = diagnose(train, pd.DataFrame({"category": ["c"]}))
    output = tmp_path / "diagnostics.json"
    assert json.loads(report.to_json(output)) == json.loads(output.read_text())
    exported = report.to_dict()
    exported["inputs"]["train"]["n_rows"] = 99
    assert report.inputs["train"]["n_rows"] == 2
    table = report.to_dataframe()
    assert set(table.section) == {"inputs", "structure", "categorical_support"}
    assert "n_unseen_cells" in set(table.check)
    assert "2 rows" in report.summary()
    assert "full training data" in report.summary()


def test_diagnostics_copies_explicit_schema_and_records_temporal_declarations():
    schema = Schema(categorical=["date"])
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"])})
    report = diagnose(frame, frame, schema=schema)
    schema.categorical.append("later")
    assert report.schema.categorical == ["date"]
    assert report.inputs["train"]["undeclared_columns"] == []
    assert report.inputs["train"]["missing_schema_columns"] == []


@pytest.mark.parametrize("role", ["train", "generated"])
def test_diagnostics_requires_both_training_and_generated_frames(role):
    frames = {"train": pd.DataFrame({"x": [1]}), "generated": pd.DataFrame({"x": [1]})}
    frames[role] = None
    with pytest.raises(TypeError, match=role):
        diagnose(**frames)
