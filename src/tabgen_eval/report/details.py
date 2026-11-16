"""Present descriptive metric arrays as named tables and JSON-safe metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from enum import Enum
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabgen_eval.execution.runtime.details import ConcentrationDetails, DetailTable, FidelityDetails
from tabgen_eval.preprocessing.categories import category_display, category_key

_LEVELS = {"column": "columns", "pair": "pairs", "slice": "slices", "radius": "radii"}


def _objects(values: Iterable[object]) -> NDArray[np.object_]:
    """Preserve scalar types and tuple labels without pandas inference."""
    values = list(values)
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        result[i] = value
    return result


def _vocabulary(metadata: Mapping[str, object], column: int) -> Sequence[object] | None:
    # This optional field is written by descriptive execution; the surrounding
    # metadata remains extensible for backend-specific scalar observations.
    values = cast(Mapping[int | str, Sequence[object]], metadata.get("category_values", {}))
    return values.get(column, values.get(str(column)))


def _labels(table: DetailTable, metadata: Mapping[str, object]) -> NDArray[np.object_]:
    result = []
    for i, (column, code) in enumerate(
        zip(table.columns["column_index"], table.columns["label_code"], strict=True)
    ):
        vocabulary = _vocabulary(metadata, int(column))
        if vocabulary is not None:
            result.append(vocabulary[int(code)])
        else:
            if not (
                np.isfinite(table.columns["lower"][i]) and np.isfinite(table.columns["upper"][i])
            ):
                raise ValueError(
                    f"Slice labels for column {int(column)} require a retained category "
                    "vocabulary or finite numeric interval bounds."
                )
            closed = (
                "both"
                if table.columns["left_closed"][i] and table.columns["right_closed"][i]
                else "left"
                if table.columns["left_closed"][i]
                else "right"
                if table.columns["right_closed"][i]
                else "neither"
            )
            result.append(
                pd.Interval(table.columns["lower"][i], table.columns["upper"][i], closed=closed)
            )
    return _objects(result)


def detail_frame(
    details: FidelityDetails | ConcentrationDetails,
    *,
    metric: str,
    level: str | None = None,
) -> pd.DataFrame:
    """Return one named detail table, independently owned by the caller.

    The default is the column table when present, then pairs or radii.
    Slice labels preserve categorical scalar identities; numeric labels are
    intervals with the conditioning bins' actual boundaries and closure.
    """
    if level is None:
        key = next(name for name in ("columns", "pairs", "radii") if name in details.tables)
    elif level in _LEVELS:
        key = _LEVELS[level]
    else:
        raise ValueError("detail level must be 'column', 'pair', 'slice', 'radius', or None")
    if key not in details.tables:
        available = [name for name, table in _LEVELS.items() if table in details.tables]
        raise ValueError(
            f"Metric {metric!r} has no {level!r} detail table; available levels: {available}"
        )
    table = details.tables[key]
    names = details.metadata.get("column_names", ())
    columns = {}
    for name, values in table.columns.items():
        if name in {"column_index", "column_1_index", "column_2_index"}:
            display = name.removesuffix("_index")
            columns[display] = _objects(names[int(index)] for index in values)
        elif name == "label_code":
            columns["label"] = _labels(table, details.metadata)
        else:
            columns[name] = values.copy()
    frame = pd.DataFrame(columns, copy=True)
    frame.attrs["metric"] = metric
    frame.attrs["level"] = next(name for name, table_name in _LEVELS.items() if table_name == key)
    frame.attrs["metadata"] = deepcopy(details.metadata)
    frame.attrs["table_metadata"] = deepcopy(table.metadata)
    return frame


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(item) for item in value]
    return str(value)


def _category_json(value: object) -> dict[str, object]:
    key = category_key(value)
    result = {
        "family": key[0] if key is not None else "missing",
        "display": _json_value(category_display(value)),
    }
    if key is not None and key[0] == "scalar":
        result["type"] = f"{type(value).__module__}.{type(value).__qualname__}"
    return result


def details_json(details: FidelityDetails | ConcentrationDetails) -> dict[str, object]:
    """Export compact column arrays with typed category vocabulary descriptions."""
    metadata = {
        key: _json_value(value)
        for key, value in details.metadata.items()
        if key != "category_values"
    }
    if "category_values" in details.metadata:
        metadata["category_values"] = {
            str(column): [_category_json(value) for value in values]
            for column, values in details.metadata["category_values"].items()
        }
    return {
        "metadata": metadata,
        "tables": {
            name: {
                "columns": {
                    column: _json_value(values) for column, values in table.columns.items()
                },
                "metadata": _json_value(table.metadata),
            }
            for name, table in details.tables.items()
        },
    }
