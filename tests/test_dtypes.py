"""Declared dtype handling without schema-driven cleaning or coercion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, ValidationError
from tabgen_eval.schema import infer_schema, validate
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit

N = 40


def _col(kind: str):
    """A length-N column of the requested dtype."""
    return {
        "datetime_naive": pd.date_range("2020-01-01", periods=N),
        "datetime_tz": pd.date_range("2020-01-01", periods=N, tz="UTC"),
        "period": pd.period_range("2020-01", periods=N, freq="M"),
        "timedelta": pd.to_timedelta(np.arange(N), unit="D"),
        "float": np.arange(N, dtype=float) / 3.0,
        "float_nullable": pd.array(np.arange(N) / 3.0, dtype="Float64"),
        "int_highcard": np.arange(N) * 7,  # > cardinality threshold -> numeric
        "int_lowcard": np.arange(N) % 3,  # low cardinality -> categorical
        "bool": np.arange(N) % 2 == 0,
        "boolean_nullable": pd.array([True, False] * (N // 2), dtype="boolean"),
        "string": pd.array([f"s{i % 4}" for i in range(N)], dtype="string"),
        "category": pd.Categorical([f"c{i % 3}" for i in range(N)]),
        "object_strings": [f"o{i % 5}" for i in range(N)],
    }[kind]


# -- inference routing ---------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected_role",
    [
        ("datetime_naive", "categorical"),
        ("datetime_tz", "categorical"),
        ("period", "categorical"),
        ("timedelta", "categorical"),
        ("float", "numeric"),
        ("float_nullable", "numeric"),
        ("int_highcard", "numeric"),
        ("int_lowcard", "categorical"),
        ("bool", "categorical"),
        ("boolean_nullable", "categorical"),
        ("string", "categorical"),
        ("category", "categorical"),
        ("object_strings", "categorical"),
    ],
)
def test_inference_routes_dtype_to_expected_role(kind, expected_role):
    df = pd.DataFrame({"x": _col(kind)})
    s = infer_schema(df)
    role = "numeric" if "x" in s.numeric else "categorical"
    assert role == expected_role


# -- evaluate never crashes on a supported dtype -------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "datetime_naive",
        "datetime_tz",
        "period",
        "timedelta",
        "float_nullable",
        "boolean_nullable",
        "string",
        "category",
    ],
)
def test_evaluate_does_not_crash_on_supported_dtype(kind):
    # Pair the column with a plain numeric so Trend has a second column to work with.
    df = pd.DataFrame({"x": _col(kind), "y": np.linspace(0, 1, N)})
    rep = evaluate(
        df,
        df.copy(),
        schema=Schema.infer(df),
        options=make_options(uncertainty="both", n_replicates=10, seed=0),
    )
    assert rep.to_dataframe().shape[0] > 0  # produced a (possibly flagged) Report


# Temporal measurements require explicit user conversion.


@pytest.mark.parametrize("kind", ["datetime_naive", "datetime_tz", "period", "timedelta"])
def test_temporal_numeric_declarations_raise_without_converting(kind):
    frame = pd.DataFrame({"x": _col(kind)})
    expected = frame.copy(deep=True)
    with pytest.raises(ValidationError, match="declared numeric"):
        validate(frame, frame, Schema(numeric=["x"]))
    pd.testing.assert_frame_equal(frame, expected)


# -- unsupported dtypes raise (fail loud at the boundary) ----------------


def test_complex_column_raises():
    df = pd.DataFrame({"x": (np.arange(N) + 1j).astype(complex)})
    with pytest.raises(ValidationError, match="unsupported dtype"):
        evaluate(
            df,
            df.copy(),
            schema=Schema.infer(df),
            options=make_options(uncertainty="both", n_replicates=5),
        )


def test_interval_column_raises():
    df = pd.DataFrame({"x": pd.IntervalIndex.from_breaks(np.arange(N + 1)).to_numpy()})
    with pytest.raises(ValidationError, match="unsupported dtype"):
        evaluate(
            df,
            df.copy(),
            schema=Schema.infer(df),
            options=make_options(uncertainty="both", n_replicates=5),
        )


def test_nested_object_column_raises():
    df = pd.DataFrame({"x": [[1, 2]] * N})  # object column holding lists
    with pytest.raises(ValidationError, match="nested"):
        evaluate(
            df,
            df.copy(),
            schema=Schema.infer(df),
            options=make_options(uncertainty="both", n_replicates=5),
        )


def test_unsupported_column_outside_schema_is_ignored():
    # A complex column the schema doesn't reference must not trip the guard.
    df = pd.DataFrame({"x": np.linspace(0, 1, N), "junk": (np.arange(N) + 1j).astype(complex)})
    schema = Schema(numeric=["x"])
    rep = evaluate(
        df, df.copy(), schema=schema, options=make_options(uncertainty="both", n_replicates=10)
    )
    assert rep.to_dataframe().shape[0] > 0
