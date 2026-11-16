"""Collect host tables from Conditional's existing slice and column reductions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tabgen_eval._typing import FloatArray
from tabgen_eval.execution.runtime.details import DetailTable, FidelityDetails

STAT_NAMES = (
    "cond_shape_all",
    "cond_shape_rare",
    "cond_trend_all",
    "cond_trend_rare",
    "recovery_all",
    "recovery_rare",
)
_SLICE_FIELDS = (
    ("column_index", np.int64),
    ("label_code", np.int64),
    ("reference_count", np.int64),
    ("candidate_count", np.int64),
    ("reference_total", np.int64),
    ("candidate_total", np.int64),
    ("eligible", bool),
    ("rare", bool),
    ("shape", float),
    ("trend", float),
    ("recovery", float),
    ("lower", float),
    ("upper", float),
    ("left_closed", bool),
    ("right_closed", bool),
)


class ConditionalDetails:
    """Retain computed values without changing slice eligibility or aggregation."""

    def __init__(
        self, column_names: Sequence[str], category_values: dict[int, tuple[object, ...]]
    ) -> None:
        self.column_names = tuple(column_names)
        self.category_values = dict(category_values)
        self.columns = np.full((len(self.column_names), len(STAT_NAMES)), np.nan)
        self.slices = []

    def add_column(self, index: int, values: Sequence[float]) -> None:
        """Store the six column means already computed by the metric."""
        self.columns[index] = values
        if len(self.column_names) < 2:
            self.columns[index, :2] = np.nan
        if len(self.column_names) < 3:
            self.columns[index, 2:4] = np.nan

    def add_slice(
        self,
        index: int,
        code: int,
        reference_count: int,
        candidate_count: int,
        reference_total: int,
        candidate_total: int,
        eligible: bool,
        rare: bool,
        shape: float,
        trend: float,
        recovery: float,
        edges: FloatArray | None,
    ) -> None:
        """Record one observed reference group and its actual interval or label ID."""
        numeric = edges is not None
        if len(self.column_names) < 2:
            shape = np.nan
        if len(self.column_names) < 3:
            trend = np.nan
        self.slices.append(
            (
                index,
                code,
                reference_count,
                candidate_count,
                reference_total,
                candidate_total,
                eligible,
                rare,
                shape,
                trend,
                recovery,
                float(edges[code]) if numeric else np.nan,
                float(edges[code + 1]) if numeric else np.nan,
                bool(numeric and code == 0),
                numeric,
            )
        )

    def finish(self) -> FidelityDetails:
        """Own equal-length arrays and attach typed labels for host presentation."""
        columns = {"column_index": np.arange(len(self.column_names), dtype=np.int64)}
        columns.update({name: self.columns[:, i].copy() for i, name in enumerate(STAT_NAMES)})
        slices = {
            name: np.asarray([row[i] for row in self.slices], dtype=dtype)
            for i, (name, dtype) in enumerate(_SLICE_FIELDS)
        }
        return FidelityDetails(
            {"columns": DetailTable(columns), "slices": DetailTable(slices)},
            {"column_names": self.column_names, "category_values": self.category_values},
        )
