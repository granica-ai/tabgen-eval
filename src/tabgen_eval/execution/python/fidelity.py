r"""Fidelity — Shape & Trend: do the synthetic marginals and pairwise relationships match?

Two complementary scores, each in ``[0, 1]`` (1 = identical), each a NaN-safe mean over
columns / column pairs (degenerate columns/pairs drop out; if *all* are degenerate the stat
is NaN, never a fake 0.0). Shape is symmetric. Trend fits numeric contingency bins
on the reference and clips candidate values to those bins, so its mixed-type
comparisons can change when the reference and candidate are exchanged.

Shape — marginal-distribution similarity, averaged over the ``p`` columns::

        Shape = (1/p) · Σ_{j=1..p} s_j

  where the per-column similarity ``s_j`` depends on the column type:

    • numeric (Kolmogorov–Smirnov complement) — 1 minus the largest gap between the two
      empirical CDFs::

            s_j = 1 − D_KS ,    D_KS = sup_x | F̂_test(x) − F̂_syn(x) |

    • categorical (total-variation complement) — over the union of category values ``c`` with
      frequencies ``p(c)``::

            s_j = 1 − ½ · Σ_c | p_test(c) − p_syn(c) |

Trend — pairwise-association similarity, averaged over the ``C(p, 2)`` column pairs::

        Trend = mean over pairs (j, k) of t_jk

  where ``t_jk`` depends on the pair's types:

    • numeric–numeric (Pearson correlation similarity) — ``|Δρ| ∈ [0, 2]`` so ``t ∈ [0, 1]``::

            t_jk = 1 − ½ · | ρ_test(j, k) − ρ_syn(j, k) |

    • otherwise — categorical–categorical, or numeric–categorical with the numeric column
      discretised into quantile bins (contingency similarity) — over the normalised joint
      contingency table of the (binned) pair::

            t_jk = 1 − ½ · Σ_{a,b} | P_test(a, b) − P_syn(a, b) |

Direction & ideal: higher-better, but the achievable target is the real-real ideal value (two real
samples differ by KS/correlation noise at finite ``n``), not 1.0. Trend needs ≥ 2 columns;
with one column it is ``NOT_APPLICABLE``. The four primitives live in ``primitives.py`` and
return NaN for unavailable comparisons, which the NaN-safe mean excludes. Correlation
is undefined for constant inputs; KS remains defined for constant columns.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, overload

import numpy as np
import pandas as pd

from tabgen_eval.execution.runtime.details import DetailedOutcome, DetailTable, FidelityDetails
from tabgen_eval.schema import Schema

from .primitives import (
    contingency_similarity,
    correlation_similarity,
    ks_complement,
    tv_complement,
)
from .stats import safe_nanmean


@overload
def compute_shape(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    details: Literal[False] = False,
) -> dict[str, float]: ...


@overload
def compute_shape(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema, *, details: Literal[True]
) -> DetailedOutcome: ...


@overload
def compute_shape(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema, *, details: bool
) -> dict[str, float] | DetailedOutcome: ...


def compute_shape(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema, *, details: bool = False
) -> dict[str, float] | DetailedOutcome:
    """Compute the finite-column mean of Python KS/TV marginal similarities.

    Read-only DataFrames follow schema roles; numeric/categorical missingness
    is handled per column by the shared reference primitives. Return the
    public statistic dictionary, with NaN when no column is computable. With
    details enabled, retain the column scores from this same computation.
    """
    scores: list[float] = []
    for col in schema.numeric:
        scores.append(ks_complement(reference[col], candidate[col]))
    for col in schema.categorical:
        scores.append(tv_complement(reference[col], candidate[col]))
    values = {"shape": safe_nanmean(scores)}
    if not details:
        return values
    return DetailedOutcome(
        values,
        FidelityDetails(
            {
                "columns": DetailTable(
                    {"column_index": np.arange(len(scores)), "score": np.asarray(scores)}
                )
            },
            {"column_names": tuple(schema.numeric) + tuple(schema.categorical)},
        ),
    )


@overload
def compute_trend(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    details: Literal[False] = False,
) -> dict[str, float]: ...


@overload
def compute_trend(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema, *, details: Literal[True]
) -> DetailedOutcome: ...


@overload
def compute_trend(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema, *, details: bool
) -> dict[str, float] | DetailedOutcome: ...


def compute_trend(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema, *, details: bool = False
) -> dict[str, float] | DetailedOutcome:
    """Compute the finite-pair mean of Python association similarities.

    Numeric pairs compare correlations; mixed/categorical pairs use reference-
    fitted contingencies. Reference order and per-pair missingness remain
    significant. Inputs are read-only and scores retain NumPy arithmetic.
    Optional pair details retain the scores used by the scalar reduction.
    """
    numeric = set(schema.numeric)
    cols = list(schema.numeric) + list(schema.categorical)
    scores: list[float] = []
    for c1, c2 in combinations(cols, 2):
        c1_num, c2_num = c1 in numeric, c2 in numeric
        if c1_num and c2_num:
            score = correlation_similarity(
                reference[c1], reference[c2], candidate[c1], candidate[c2]
            )
        else:
            score = contingency_similarity(
                reference[c1],
                reference[c2],
                candidate[c1],
                candidate[c2],
                col1_numeric=c1_num,
                col2_numeric=c2_num,
            )
        scores.append(score)
    values = {"trend": safe_nanmean(scores)}
    if not details:
        return values
    first, second = np.triu_indices(len(cols), k=1)
    return DetailedOutcome(
        values,
        FidelityDetails(
            {
                "pairs": DetailTable(
                    {
                        "column_1_index": first,
                        "column_2_index": second,
                        "score": np.asarray(scores),
                    }
                )
            },
            {"column_names": tuple(cols)},
        ),
    )
