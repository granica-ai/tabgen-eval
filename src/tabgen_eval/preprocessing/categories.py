"""Typed categorical identities shared by preparation and fitted encoders.

String contents are exact. Integers, floats, booleans and strings occupy separate
families, with Python and NumPy scalars sharing their logical family. Actual
nulls have no identity; the explicit missing category is an ordinary observation.
This module neither selects rows nor fits any numerical transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tabgen_eval._typing import IntArray

CategoryKey: TypeAlias = tuple[object, ...]
CategoryColumn: TypeAlias = pd.Series | ArrayLike


class _MissingCategory(Enum):
    """Pickle-safe identity for an explicitly observed missing predictor."""

    VALUE = "missing-category"


MISSING_CATEGORY = _MissingCategory.VALUE


def category_key(value: object) -> CategoryKey | None:
    """Return a hashable typed identity, or ``None`` for an actual scalar null.

    Standard temporal, byte, decimal and tuple labels retain their own families.
    Other hashable scalar types retain their concrete type and native equality.
    Objects without scalar equality or hashing must be converted by the caller.
    """
    if value is MISSING_CATEGORY:
        return ("missing-category",)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if isinstance(value, (str, np.str_)):
        return ("string", str.__str__(value))
    if isinstance(value, (bool, np.bool_)):
        return ("boolean", bool(value))
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        return ("datetime", pd.Timestamp(value))
    if isinstance(value, (timedelta, pd.Timedelta, np.timedelta64)):
        return ("timedelta", pd.Timedelta(value))
    if isinstance(value, (int, np.integer)):
        return ("integer", int(value))
    if isinstance(value, (float, np.floating)):
        # NumPy longdouble may contain precision beyond a Python float. Numeric
        # scalar hashing/equality already agrees across equal float values.
        if isinstance(value, np.longdouble):
            number = np.longdouble(value)
        elif isinstance(value, float):
            number = float.__float__(value)
        else:
            number = float(value)
        return ("float", number)
    if isinstance(value, (bytes, np.bytes_)):
        return ("bytes", bytes(value))
    if isinstance(value, date):
        return ("date", value)
    if isinstance(value, Decimal):
        return ("decimal", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(category_key(item) for item in value))
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("Categorical labels must be hashable scalars.") from error
    return ("scalar", type(value), value)


def _local_categories(
    values: CategoryColumn,
) -> tuple[IntArray, tuple[CategoryKey | None, ...], tuple[object, ...]]:
    """Factor one column, visiting repeated homogeneous labels only once."""
    if isinstance(values, pd.Series):
        series = values
    else:
        # A Python sequence can contain distinct integer and float labels.
        # Preserve those types before pandas has an opportunity to infer float.
        series = pd.Series(values, dtype=object if isinstance(values, (list, tuple)) else None)
    dtype = series.dtype
    # Object string columns are common in Parquet-backed benchmarks. Arrow
    # classifies/encodes their values in compiled code, including nulls, so a
    # Python scalar-type scan is unnecessary. Mixed scalar families still use
    # the exact typed path below and never merge True, 1 and 1.0.
    if pd.api.types.is_object_dtype(dtype) and pd.api.types.infer_dtype(series, skipna=True) in {
        "string",
        "unicode",
    }:
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
        except ModuleNotFoundError as error:
            if error.name != "pyarrow":
                raise
        else:
            try:
                encoded = pc.dictionary_encode(pa.array(series, from_pandas=True))
                labels = tuple(encoded.dictionary.to_pylist())
                codes = pc.fill_null(encoded.indices, -1).to_numpy(zero_copy_only=False)
                return codes, tuple(category_key(value) for value in labels), labels
            except (
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
                UnicodeEncodeError,
            ):
                pass
    # Object equality merges True, 1 and 1.0 before a typed key can recover them.
    # Extension categorical dictionaries may likewise mix scalar families, so
    # inspect their dictionary rather than their expanded rows.
    extended_float = isinstance(dtype, np.dtype) and dtype.kind == "f" and dtype.itemsize > 8
    homogeneous = not pd.api.types.is_object_dtype(dtype) and not extended_float
    if (
        not homogeneous
        and not extended_float
        and pd.api.types.infer_dtype(series, skipna=True)
        in {
            "string",
            "unicode",
            "bytes",
            "integer",
            "floating",
            "boolean",
            "empty",
        }
    ):
        # Scalar subclasses can override equality. Only builtin/NumPy scalar
        # implementations may be grouped before their typed key is computed.
        ordinary = {str, bytes, int, float, bool, type(None), type(pd.NA), type(pd.NaT)}
        homogeneous = all(
            type(value) in ordinary or type(value).__module__ == "numpy"
            for value in series.to_numpy()
        )
    if homogeneous:
        raw_codes, labels = pd.factorize(series, sort=False)
        labels = tuple(labels)
    else:
        keys, labels, lookup = [], [], {}
        codes = np.empty(len(series), dtype=np.int64)
        for i, value in enumerate(series.to_numpy()):
            key = category_key(value)
            if key is None:
                codes[i] = -1
                continue
            code = lookup.get(key)
            if code is None:
                code = len(keys)
                lookup[key] = code
                keys.append(key)
                labels.append(value)
            codes[i] = code
        return codes, tuple(keys), tuple(labels)
    keys = tuple(category_key(value) for value in labels)
    return raw_codes, keys, labels


def category_display(value: object) -> str | bool | int | float | None:
    """Return a JSON-safe diagnostic label; identity never depends on display."""
    if value is MISSING_CATEGORY:
        return "<missing category>"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    return str(value)


@dataclass(frozen=True)
class CategoryVocabulary:
    """Observed typed labels in first-appearance order.

    ``transform`` returns ``-1`` for actual nulls and ``-2`` for unseen observed
    labels. A fitted vocabulary never grows while transforming scoring rows.
    """

    keys: tuple[CategoryKey | None, ...]
    values: tuple[object, ...]

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def fit(cls, values: CategoryColumn) -> CategoryVocabulary:
        """Fit identities using only the supplied training column."""
        return factorize_categories(values)[1]

    def transform(self, values: CategoryColumn) -> IntArray:
        """Encode a column without learning any labels from it."""
        codes, keys, _ = _local_categories(values)
        lookup = {key: i for i, key in enumerate(self.keys)}
        mapping = np.array([lookup.get(key, -2) for key in keys], dtype=np.int64)
        result = np.full(len(codes), -1, dtype=np.int64)
        observed = codes >= 0
        result[observed] = mapping[codes[observed]]
        return result


def factorize_categories(
    *columns: CategoryColumn,
) -> tuple[tuple[IntArray, ...], CategoryVocabulary]:
    """Encode columns into one vocabulary without changing row multiplicities.

    Return ``(code_arrays, vocabulary)``. Each int64 array follows its input row
    order and uses ``-1`` for actual nulls. Sharing identities is a storage step;
    callers fitting a model must instead fit a training-only vocabulary.
    """
    lookup, keys, labels, results = {}, [], [], []
    for column in columns:
        local, local_keys, local_labels = _local_categories(column)
        mapping = np.empty(len(local_keys), dtype=np.int64)
        for i, (key, value) in enumerate(zip(local_keys, local_labels, strict=True)):
            if key is None:
                mapping[i] = -1
                continue
            code = lookup.get(key)
            if code is None:
                code = len(keys)
                lookup[key] = code
                keys.append(key)
                labels.append(value)
            mapping[i] = code
        result = np.full(len(local), -1, dtype=np.int64)
        observed = local >= 0
        result[observed] = mapping[local[observed]]
        results.append(result)
    return tuple(results), CategoryVocabulary(tuple(keys), tuple(labels))
