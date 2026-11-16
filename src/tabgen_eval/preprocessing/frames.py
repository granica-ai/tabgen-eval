"""Host-frame adapters for independent classifier preprocessing checks."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from tabgen_eval._typing import FloatArray, Precision
from tabgen_eval.schema import Schema

from .folds import fit_fold
from .sources import prepare_sources


def fold_design(
    reference_train: pd.DataFrame,
    candidate_train: pd.DataFrame,
    reference_test: pd.DataFrame,
    candidate_test: pd.DataFrame,
    schema: Schema,
    limit: float,
    *,
    policy: Literal["linear", "xgboost"] = "linear",
    dtype: Precision = "float64",
) -> tuple[FloatArray, ...]:
    """Build a dense design through shared compiled preprocessing.

    Tests use this adapter to inspect dense designs and compare independent
    reference solvers. Production classifiers reuse source storage across folds.
    Model-specific solver workspace remains the solver's responsibility.
    """
    reference = pd.concat((reference_train, reference_test), ignore_index=True)
    candidate = pd.concat((candidate_train, candidate_test), ignore_index=True)
    sources = prepare_sources(reference, candidate, schema, dtype=dtype)
    rf = np.repeat([1, 0], [len(reference_train), len(reference_test)])
    sf = np.repeat([1, 0], [len(candidate_train), len(candidate_test)])
    data = fit_fold(
        sources.reference,
        sources.candidate,
        rf,
        sf,
        0,
        cardinalities=sources.cardinalities,
        numeric_names=sources.numeric_names,
        categorical_names=sources.categorical_names,
        category_values=sources.category_values,
        policy=policy,
        memory_limit_bytes=limit,
    )
    return tuple(
        value.numpy()
        for value in (data.train, data.test, data.training_labels, data.scoring_labels)
    )
