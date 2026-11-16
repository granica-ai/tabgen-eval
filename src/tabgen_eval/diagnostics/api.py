"""Standalone data diagnostics against the generator's actual training data."""

from __future__ import annotations

import pandas as pd

from .._validation import require_instance
from ..schema import Schema
from . import profiling
from .report import DiagnosticReport


def diagnose(
    train: pd.DataFrame, generated: pd.DataFrame, *, schema: Schema | None = None
) -> DiagnosticReport:
    """Describe raw inputs and generated support relative to full training data.

    This workflow performs no metric evaluation, cleaning, row sampling, model
    fitting, or uncertainty quantification. Schema and dtype differences are
    observations in the report. Comparisons requiring absent or ambiguous
    columns are omitted; their structure differences remain visible.
    """
    for role, frame in (("train", train), ("generated", generated)):
        require_instance(frame, pd.DataFrame, f"{role} must be a pandas DataFrame")
    if schema is not None:
        require_instance(schema, Schema, "schema must be a Schema or None")
    if schema is None:
        schema = Schema.infer(train.loc[:, ~train.columns.duplicated()])
    else:
        schema = Schema.from_dict(schema.to_dict())
    return DiagnosticReport(
        schema=schema,
        inputs={
            role: profiling.profile(frame, schema)
            for role, frame in (("train", train), ("generated", generated))
        },
        structure=profiling.structure(train, generated),
        categorical_support=profiling.categorical_support(train, generated, schema),
        numeric_support=profiling.numeric_support(train, generated, schema),
    )
