"""Full-input descriptive profiles and support comparisons against training."""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

import numpy as np
import pandas as pd

from .._typing import FloatArray
from ..preprocessing.categories import (
    CategoryVocabulary,
    category_display,
    category_key,
    factorize_categories,
)
from ..schema import Schema


class ColumnProfile(TypedDict):
    """One raw input column's position, storage dtype, and missing count."""

    name: object
    position: int
    dtype: str
    n_missing: int


class InputProfile(TypedDict):
    """Full-frame observations before any metric-specific row exclusions."""

    n_rows: int
    n_columns: int
    n_missing_cells: int
    missing_cell_rate: float | None
    n_missing_rows: int
    missing_row_rate: float | None
    n_duplicate_rows: int
    duplicate_row_rate: float | None
    missing_schema_columns: list[str]
    undeclared_columns: list[object]
    columns: list[ColumnProfile]


class StructureObservations(TypedDict):
    """Column identity, ordering and storage differences between raw inputs."""

    missing_columns: list[object]
    unexpected_columns: list[object]
    column_order_matches: bool
    duplicate_train_columns: list[object]
    duplicate_generated_columns: list[object]
    dtype_mismatches: dict[object, dict[str, str]]


class CategoryObservation(TypedDict):
    type: object
    value: str | bool | int | float | None


class CategoricalSupport(TypedDict):
    """Observed label support relative to the entire training pool."""

    n_train_categories: int
    n_generated_categories: int
    n_unseen_categories: int
    n_unseen_cells: int
    unseen_cell_rate: float | None
    n_train_categories_absent: int
    unseen_categories: list[CategoryObservation]


class NumericSupport(TypedDict):
    """Observed ranges and unsupported numeric values in each raw source."""

    train_min: float | None
    train_max: float | None
    generated_min: float | None
    generated_max: float | None
    n_below_train_min: int | None
    n_above_train_max: int | None
    n_train_non_numeric: int
    n_generated_non_numeric: int
    n_train_nonfinite: int
    n_generated_nonfinite: int


def _rate(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def profile(frame: pd.DataFrame, schema: Schema) -> InputProfile:
    """Observe a raw input without cleaning, sampling, or quality judgments."""
    missing = frame.isna()
    n_missing_cells = int(missing.to_numpy().sum())
    n_missing_rows = int(missing.any(axis=1).sum())
    # Typed category identities prevent pandas merging True, 1, and 1.0 in
    # mixed object columns when determining exact duplicate records.
    encoded = pd.DataFrame(
        {
            position: factorize_categories(frame.iloc[:, position])[0][0]
            for position in range(frame.shape[1])
        },
        index=frame.index,
    )
    n_duplicate_rows = int(encoded.duplicated().sum()) if frame.shape[1] else 0
    declared = set(schema.numeric + schema.categorical)
    if schema.target is not None:
        declared.add(schema.target)
    return {
        "n_rows": len(frame),
        "n_columns": frame.shape[1],
        "n_missing_cells": n_missing_cells,
        "missing_cell_rate": _rate(n_missing_cells, frame.size),
        "n_missing_rows": n_missing_rows,
        "missing_row_rate": _rate(n_missing_rows, len(frame)),
        "n_duplicate_rows": n_duplicate_rows,
        "duplicate_row_rate": _rate(n_duplicate_rows, len(frame)),
        "missing_schema_columns": sorted(declared - set(frame.columns)),
        "undeclared_columns": [column for column in frame.columns if column not in declared],
        "columns": [
            {
                "name": column,
                "position": position,
                "dtype": str(frame.iloc[:, position].dtype),
                "n_missing": int(missing.iloc[:, position].sum()),
            }
            for position, column in enumerate(frame.columns)
        ],
    }


def structure(train: pd.DataFrame, generated: pd.DataFrame) -> StructureObservations:
    """Observe column and dtype differences, including ambiguous duplicate names."""
    train_counts, generated_counts = Counter(train.columns), Counter(generated.columns)
    common = [
        column for column in train.columns if train_counts[column] == generated_counts[column] == 1
    ]
    return {
        "missing_columns": [column for column in train.columns if column not in generated.columns],
        "unexpected_columns": [
            column for column in generated.columns if column not in train.columns
        ],
        "column_order_matches": list(train.columns) == list(generated.columns),
        "duplicate_train_columns": list(train.columns[train.columns.duplicated()]),
        "duplicate_generated_columns": list(generated.columns[generated.columns.duplicated()]),
        "dtype_mismatches": {
            column: {"train": str(train[column].dtype), "generated": str(generated[column].dtype)}
            for column in common
            if train[column].dtype != generated[column].dtype
        },
    }


def categorical_support(
    train: pd.DataFrame, generated: pd.DataFrame, schema: Schema
) -> dict[str, CategoricalSupport]:
    """Compare observed typed categories to the entire original training input."""
    out = {}
    for column in schema.categorical:
        if not _shared_column(train, generated, column):
            continue
        training = CategoryVocabulary.fit(train[column])
        observed = CategoryVocabulary.fit(generated[column])
        codes = training.transform(generated[column])
        training_keys = set(training.keys)
        generated_keys = set(observed.keys)
        unseen = [
            value
            for key, value in zip(observed.keys, observed.values, strict=True)
            if key not in training_keys
        ]
        n_unseen = int((codes == -2).sum())
        n_observed = int((codes != -1).sum())
        out[column] = {
            "n_train_categories": len(training),
            "n_generated_categories": len(observed),
            "n_unseen_categories": len(unseen),
            "n_unseen_cells": n_unseen,
            "unseen_cell_rate": _rate(n_unseen, n_observed),
            "n_train_categories_absent": len(training_keys - generated_keys),
            "unseen_categories": [
                {"type": category_key(value)[0], "value": category_display(value)}
                for value in unseen
            ],
        }
    return out


def _shared_column(train: pd.DataFrame, generated: pd.DataFrame, column: str) -> bool:
    return isinstance(train.get(column), pd.Series) and isinstance(generated.get(column), pd.Series)


def _numeric_values(values: pd.Series) -> tuple[FloatArray, int, int]:
    converted = pd.to_numeric(values, errors="coerce")
    numbers = converted.to_numpy(dtype=np.float64, na_value=np.nan)
    n_non_numeric = int((converted.isna() & values.notna()).sum())
    n_nonfinite = int((~np.isfinite(numbers) & ~np.isnan(numbers)).sum())
    return numbers[np.isfinite(numbers)], n_non_numeric, n_nonfinite


def numeric_support(
    train: pd.DataFrame, generated: pd.DataFrame, schema: Schema
) -> dict[str, NumericSupport]:
    """Observe numeric ranges and conversion failures without altering inputs."""
    out = {}
    for column in schema.numeric:
        if not _shared_column(train, generated, column):
            continue
        training, train_invalid, train_nonfinite = _numeric_values(train[column])
        observed, generated_invalid, generated_nonfinite = _numeric_values(generated[column])
        lower = float(training.min()) if len(training) else None
        upper = float(training.max()) if len(training) else None
        out[column] = {
            "train_min": lower,
            "train_max": upper,
            "generated_min": float(observed.min()) if len(observed) else None,
            "generated_max": float(observed.max()) if len(observed) else None,
            "n_below_train_min": int((observed < lower).sum()) if lower is not None else None,
            "n_above_train_max": int((observed > upper).sum()) if upper is not None else None,
            "n_train_non_numeric": train_invalid,
            "n_generated_non_numeric": generated_invalid,
            "n_train_nonfinite": train_nonfinite,
            "n_generated_nonfinite": generated_nonfinite,
        }
    return out
