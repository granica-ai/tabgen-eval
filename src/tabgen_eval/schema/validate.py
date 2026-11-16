"""Validate declared representations without converting or repairing input values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from tabgen_eval._typing import FloatArray

from .schema import Schema

_NESTED_TYPES = (list, dict, set, frozenset, np.ndarray)


class ValidationError(ValueError):
    """A supplied table does not satisfy its declared input contract."""


def _require_dataframe(obj: object, name: str) -> None:
    if not isinstance(obj, pd.DataFrame):
        raise ValidationError(f"`{name}` must be a pandas.DataFrame, got {type(obj).__name__}.")
    if not obj.columns.is_unique:
        raise ValidationError(f"`{name}` contains duplicate column names.")


def numeric_array(series: pd.Series, *, role: str | None = None) -> FloatArray:
    """Extract real numeric storage, preserving NaNs and representable FP32 values.

    Numeric text, object storage, booleans, temporal values, complex values and
    infinities require explicit user preparation. Nullable numeric dtypes are
    accepted; their missing cells become NumPy NaNs for metric computations.
    """
    location = f"Column {series.name!r}" + (f" in `{role}`" if role is not None else "")
    if (
        not pdt.is_numeric_dtype(series.dtype)
        or pdt.is_bool_dtype(series.dtype)
        or pdt.is_complex_dtype(series.dtype)
        or pdt.is_timedelta64_dtype(series.dtype)
    ):
        raise ValidationError(
            f"{location} is declared numeric but has dtype {series.dtype!s}. "
            "Supply real numeric values; convert text or temporal measurements "
            "to numeric values in chosen units before evaluating."
        )
    storage = getattr(series.dtype, "numpy_dtype", series.dtype)
    try:
        storage = np.dtype(storage)
    except TypeError:
        storage = np.dtype("float64")
    dtype = np.float32 if storage.kind == "f" and storage.itemsize <= 4 else np.float64
    with np.errstate(over="ignore", invalid="ignore"):
        values = series.to_numpy(dtype=dtype, na_value=np.nan)
    if np.isinf(values).any():
        raise ValidationError(
            f"{location} contains numeric infinity or values outside the supported "
            "numeric range. Prepare finite values or genuine missing cells before evaluating."
        )
    return values


def validate(
    holdout: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    allow_empty: bool = False,
    reference_name: str = "holdout",
    candidate_name: str = "generated",
) -> None:
    """Validate table structure and every declared column without modifying inputs.

    Genuine missing cells are accepted. Their treatment belongs to each metric:
    observed-value fidelity, complete-case distance geometry, and predictor/target
    model preparation have different contracts.
    """
    if not isinstance(schema, Schema):
        raise TypeError("schema must be an explicit Schema")
    _require_dataframe(holdout, reference_name)
    _require_dataframe(generated, candidate_name)
    for name, frame in ((reference_name, holdout), (candidate_name, generated)):
        if not allow_empty and len(frame) == 0:
            raise ValidationError(f"`{name}` is empty (0 rows).")

    holdout_cols = set(holdout.columns)
    generated_cols = set(generated.columns)
    if holdout_cols != generated_cols:
        missing = sorted(holdout_cols - generated_cols)
        extra = sorted(generated_cols - holdout_cols)
        raise ValidationError(
            f"`{reference_name}` and `{candidate_name}` must have the same columns. "
            f"missing_in_{candidate_name}={missing}, extra_in_{candidate_name}={extra}. "
            "We align column order automatically but never guess a column mapping."
        )
    absent = sorted(set(schema.columns) - holdout_cols)
    if absent:
        raise ValidationError(f"Schema declares columns not present in the data: {absent}.")

    _check_supported_dtypes(holdout, reference_name, schema)
    if generated is not holdout:
        _check_supported_dtypes(generated, candidate_name, schema)


def _check_supported_dtypes(df: pd.DataFrame, name: str, schema: Schema) -> None:
    """Validate used columns only; unused values cannot change metric eligibility."""
    for col in schema.columns:
        series = df[col]
        if col in schema.numeric or (
            col == schema.target
            and col not in schema.categorical
            and pdt.is_numeric_dtype(series.dtype)
            and not pdt.is_bool_dtype(series.dtype)
        ):
            numeric_array(series, role=name)
        kind = _unsupported_kind(series)
        if kind is not None:
            raise ValidationError(
                f"Column {col!r} in `{name}` has unsupported dtype ({kind}). "
                "Supply scalar numeric or categorical values, or exclude the column, "
                "before evaluating."
            )


def _unsupported_kind(series: pd.Series) -> str | None:
    """Return the unsupported representation, checking all object values."""
    if pdt.is_complex_dtype(series):
        return f"complex ({series.dtype})"
    if isinstance(series.dtype, pd.IntervalDtype):
        return "interval"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return _unsupported_kind(pd.Series(series.cat.categories))
    if pdt.is_object_dtype(series):
        inferred = pdt.infer_dtype(series, skipna=True)
        if inferred in {
            "string",
            "unicode",
            "bytes",
            "integer",
            "floating",
            "boolean",
            "empty",
            "datetime",
            "datetime64",
            "date",
            "timedelta",
            "timedelta64",
            "period",
            "decimal",
        }:
            return None
        for value in series.to_numpy(copy=False):
            if _nested(value):
                return "object column with nested (list/dict/set/array) values"
            if isinstance(value, (complex, np.complexfloating)):
                return "complex object values"
            if isinstance(value, pd.Interval):
                return "interval object values"
    return None


def _nested(value: object) -> bool:
    """Tuples can be scalar category identities, provided their members are scalar."""
    return isinstance(value, _NESTED_TYPES) or (
        isinstance(value, tuple) and any(_nested(item) for item in value)
    )
