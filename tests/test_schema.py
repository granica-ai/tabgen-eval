"""Schema inference + validation: types, column agreement, tiny/empty frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, ValidationError
from tabgen_eval.schema import infer_schema, validate

pytestmark = pytest.mark.unit


def test_infer_basic_roles():
    df = pd.DataFrame(
        {
            "x": np.arange(100.0),  # float -> numeric
            "k": np.arange(100),  # high-card int -> numeric
            "flag": [True, False] * 50,  # bool -> categorical
            "cat": ["a", "b"] * 50,  # object -> categorical
        }
    )
    s = infer_schema(df)
    assert "x" in s.numeric
    assert "k" in s.numeric
    assert "flag" in s.categorical
    assert "cat" in s.categorical


def test_infer_low_cardinality_int_is_categorical():
    df = pd.DataFrame({"c": [0, 1, 2, 0, 1, 2] * 10})
    assert "c" in infer_schema(df).categorical


def test_infer_int_coded_categorical_with_nan_stays_categorical():
    # A missing value forces an integer-coded category to float dtype; it must still infer as
    # categorical, not flip to numeric just because of one NaN.
    df = pd.DataFrame({"code": [1.0, 2.0, 3.0, np.nan, 1.0, 2.0] * 5})
    s = infer_schema(df)
    assert "code" in s.categorical and "code" not in s.numeric


def test_infer_continuous_float_stays_numeric():
    # Non-integer values stay numeric even at low cardinality...
    df = pd.DataFrame({"x": [0.5, 1.5, 2.25, 0.5, 1.5] * 10})
    assert "x" in infer_schema(df).numeric
    # ...and a high-cardinality integer-valued float is a measurement, not a code.
    df2 = pd.DataFrame({"y": np.arange(100.0)})
    assert "y" in infer_schema(df2).numeric


def test_schema_rejects_overlap():
    with pytest.raises(ValueError):
        Schema(numeric=["a"], categorical=["a"])


@pytest.mark.parametrize("role", ["numeric", "categorical"])
def test_schema_rejects_duplicate_entries(role):
    with pytest.raises(ValueError, match="Duplicate"):
        Schema(**{role: ["a", "a"]})


@pytest.mark.parametrize("retired", ["integer", "datetime"])
def test_schema_rejects_retired_hints(retired):
    with pytest.raises(TypeError):
        Schema(**{retired: ["a"]})
    with pytest.raises(TypeError):
        Schema.from_dict({retired: ["a"]})


def test_validate_rejects_absent_declared_column():
    schema = Schema(numeric=["a", "t"])
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValidationError, match="t"):
        validate(frame, frame, schema)


def test_validate_rejects_non_dataframe():
    s = Schema(numeric=["a"])
    with pytest.raises(ValidationError):
        validate(np.zeros((3, 1)), pd.DataFrame({"a": [1]}), s)


def test_validate_rejects_empty():
    s = Schema(numeric=["a"])
    with pytest.raises(ValidationError):
        validate(pd.DataFrame({"a": []}), pd.DataFrame({"a": [1]}), s)


def test_validate_rejects_column_mismatch():
    s = Schema(numeric=["a"])
    test = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    syn = pd.DataFrame({"a": [1, 2], "c": [3, 4]})
    with pytest.raises(ValidationError) as exc:
        validate(test, syn, s)
    assert "extra_in_generated" in str(exc.value) or "missing_in_generated" in str(exc.value)


def test_validate_accepts_reordered_columns():
    s = Schema(numeric=["a", "b"])
    test = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    syn = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
    validate(test, syn, s)  # no raise
