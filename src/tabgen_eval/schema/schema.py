"""Explicit column roles; callers own the interpretation of their values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd

DEFAULT_CATEGORY_THRESHOLD = 20


class SchemaDeclaration(TypedDict, total=False):
    """Serializable column roles; omitted fields use ``Schema`` defaults."""

    numeric: list[str]
    categorical: list[str]
    target: str | None


@dataclass
class Schema:
    """Declare numeric measurements, categorical labels, and an optional target.

    Evaluation requires an explicit schema and never changes values to make them
    match a declaration. Numeric measurements must already have real numeric
    storage; convert temporal measurements to chosen units before evaluation.
    Categorical labels retain scalar types and exact string contents. Missing
    values follow each metric's documented contract.

    ``Schema.infer`` is an optional proposal helper. Callers must inspect its
    result and are responsible for accepting or correcting the inferred roles.
    Columns omitted from both feature roles do not participate in feature metrics.
    ``target`` is required only for utility and may also have a feature role.
    """

    numeric: list[str] = field(default_factory=list[str])
    categorical: list[str] = field(default_factory=list[str])
    target: str | None = None

    def __post_init__(self) -> None:
        self.numeric = list(self.numeric)
        self.categorical = list(self.categorical)
        for role in ("numeric", "categorical"):
            columns = getattr(self, role)
            if len(columns) != len(set(columns)):
                raise ValueError(f"Duplicate columns in Schema.{role}")
        overlap = set(self.numeric) & set(self.categorical)
        if overlap:
            raise ValueError(f"Columns declared as both numeric and categorical: {sorted(overlap)}")

    @property
    def columns(self) -> list[str]:
        """All declared columns, numeric first then categorical, then target."""
        columns = list(self.numeric) + list(self.categorical)
        if self.target is not None and self.target not in columns:
            columns.append(self.target)
        return columns

    @classmethod
    def infer(
        cls,
        df: pd.DataFrame,
        *,
        target: str | None = None,
        int_as_categorical_max_unique: int = DEFAULT_CATEGORY_THRESHOLD,
    ) -> Schema:
        """Propose roles from dtypes and cardinality for the caller to review."""
        from .infer import infer_schema

        return infer_schema(
            df,
            target=target,
            int_as_categorical_max_unique=int_as_categorical_max_unique,
        )

    def to_dict(self) -> SchemaDeclaration:
        """Serialize the current declaration."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, declaration: SchemaDeclaration) -> Schema:
        """Construct the current schema format, rejecting unknown fields."""
        return cls(**declaration)
