"""Schema inference from dtypes + cardinality heuristics.

This helper proposes roles for callers to inspect and accept explicitly. Evaluation
never invokes it automatically and never converts values to match inferred roles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .schema import DEFAULT_CATEGORY_THRESHOLD, Schema

# Only disambiguates INTEGER columns: an integer column with at most this many distinct
# values is guessed to be category codes rather than a measurement.
# Kept low so genuine numeric measurements (age, counts) are not mis-cast as categorical;
# raise it only if your integer category codes have many levels (or just pass a Schema).


def infer_schema(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    int_as_categorical_max_unique: int = DEFAULT_CATEGORY_THRESHOLD,
) -> Schema:
    """Infer column roles from *df*.

    Heuristics:
        - float columns -> numeric, UNLESS every finite value is integer-valued and there are
          few distinct ones (<= ``int_as_categorical_max_unique``): then categorical. A missing
          value forces an integer-coded category column to float dtype (e.g. ``[1, 2, NaN, 3]``);
          without this rule a single NaN would silently flip its inferred role to numeric.
        - bool / object / string / pandas categorical -> categorical (no cardinality limit).
        - integer columns -> categorical if they have few distinct values
          (<= ``int_as_categorical_max_unique``), else numeric. This threshold only
          disambiguates integers; it never caps categorical columns.
        - datetime / period / timedelta -> categorical labels. For temporal
          measurements, convert values to numeric units and declare numeric explicitly.

    The ``target`` (if given) keeps whatever role its dtype implies but is recorded.
    """
    numeric: list[str] = []
    categorical: list[str] = []

    for col in df.columns:
        s = df[col]
        if (
            pdt.is_datetime64_any_dtype(s)
            or isinstance(s.dtype, pd.PeriodDtype)
            or pdt.is_timedelta64_dtype(s)
        ):
            categorical.append(col)
        elif pdt.is_bool_dtype(s):
            categorical.append(col)
        elif pdt.is_float_dtype(s):
            vals = s.to_numpy(dtype=float, na_value=np.nan)
            finite = vals[np.isfinite(vals)]
            if (
                finite.size > 0
                and pd.Series(finite).nunique() <= int_as_categorical_max_unique
                and np.array_equal(finite, np.round(finite))
            ):
                # Integer-coded category whose missing values forced a float dtype.
                categorical.append(col)
            else:
                numeric.append(col)
        elif pdt.is_integer_dtype(s):
            if s.nunique(dropna=True) <= int_as_categorical_max_unique:
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            # object / string / category — treat as categorical labels.
            categorical.append(col)

    return Schema(
        numeric=numeric,
        categorical=categorical,
        target=target,
    )
