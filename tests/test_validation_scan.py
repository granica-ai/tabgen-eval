"""Input validation checks all used object values without modifying source data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, ValidationError
from tabgen_eval.schema import validate
from tabgen_eval.schema.validate import _unsupported_kind

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("nested", [[1], {"key": 1}, {1}, np.array([1, 2])])
@pytest.mark.parametrize("position", [0, 49, 50, 100000])
def test_nested_values_are_rejected_at_every_position(nested, position):
    series = pd.Series([None] * 1000 + ["valid"] * position + [nested], dtype=object)
    assert "nested" in _unsupported_kind(series)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [None, np.nan, pd.NA, pd.NaT] * 1000,
        ["None", "nan", "<NA>", "NaT", "", False, 0, 1.5, (1, 2)],
    ],
)
def test_scalar_labels_and_missing_cells_are_supported(values):
    assert _unsupported_kind(pd.Series(values, dtype=object)) is None


@pytest.mark.parametrize("multiindex", [False, True])
def test_scan_is_positional_and_preserves_input(multiindex):
    values = [None] * 1300 + ["valid"] * 100 + [{"bad": 1}, "after"]
    index = (
        pd.MultiIndex.from_arrays([["duplicate"] * len(values), [7] * len(values)])
        if multiindex
        else pd.Index(["duplicate"] * len(values))
    )
    series = pd.Series(values, index=index, name="labels", dtype=object)
    series.attrs["source"] = "unchanged"
    expected = series.copy(deep=True)
    assert "nested" in _unsupported_kind(series)
    pd.testing.assert_series_equal(series, expected)
    assert series.attrs == expected.attrs


def test_nested_input_error_identifies_column_and_pool():
    schema = Schema(categorical=["labels"])
    real = pd.DataFrame({"labels": ["valid"] * 100})
    fake = pd.DataFrame({"labels": [None] * 1000 + ["valid"] * 100 + [[1, 2]]})
    with pytest.raises(ValidationError, match="Column 'labels' in `generated`.*nested"):
        validate(real, fake, schema)


def test_nested_tuple_members_are_rejected():
    frame = pd.DataFrame({"labels": [(1, [2])]})
    with pytest.raises(ValidationError, match="nested"):
        validate(frame, frame, Schema(categorical=["labels"]))


def test_excluded_nested_values_do_not_affect_validation():
    frame = pd.DataFrame({"x": [1.0], "ignored": [[1, 2]]})
    validate(frame, frame, Schema(numeric=["x"]))
