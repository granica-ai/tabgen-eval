"""Shape and Trend distribution-similarity metrics.

Shape compares column marginals; Trend compares pairwise associations. Both
average usable similarities with equal weight and report scores in [0, 1]."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabgen_eval.execution.python.fidelity import compute_shape, compute_trend
from tabgen_eval.execution.runtime.dispatch import compute_metric

from ..schema import Schema
from .base import Direction, Metric, MetricFamily, StatSpec


class Shape(Metric):
    """Average marginal-distribution similarity across declared columns.

    Numeric columns use one minus the two-sample KS statistic; categorical
    columns use one minus total variation. The ``shape`` score is in [0, 1]
    and higher is better. Each column contributes equally; missing cells are
    excluded within that column and undefined columns do not enter the mean.
    Shared execution options are documented on Metric; there are no additional
    Shape-specific statistical tunables.
    """

    name = "shape"
    category = "fidelity"
    family = MetricFamily.SAMPLE_BASED
    symmetric = True
    stats = (StatSpec("shape", "Shape", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),)

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> dict[str, float]:
        if (out := compute_metric(self, reference, candidate, schema)) is not None:
            return out
        return compute_shape(reference, candidate, schema)


class Trend(Metric):
    """Average pairwise-association similarity across declared column pairs.

    Numeric pairs compare Pearson correlations; pairs containing categoricals
    compare contingency distributions, with numeric bins fitted to the reference.
    The ``trend`` score is in [0, 1] and higher is better. Each usable pair has
    equal weight, and missing cells are excluded pairwise. At least two declared
    columns are required. Reference-anchored binning makes mixed/categorical
    Trend directional, so swapping real and generated inputs can change it.
    Shared execution options are documented on Metric; no additional public
    Trend-specific tuning parameters are required.
    """

    name = "trend"
    category = "fidelity"
    family = MetricFamily.SAMPLE_BASED
    # Contingency binning is anchored to the reference (real) range and the candidate is clipped
    # to it, so the cat/num-cat part is directional — Trend is not strictly symmetric. (The
    # num-num correlation part is symmetric.)
    symmetric = False
    stats = (StatSpec("trend", "Trend", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),)

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        # Trend needs at least two columns to form a pair.
        if len(schema.numeric) + len(schema.categorical) < 2:
            return {"trend"}
        return set()

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> dict[str, float]:
        if (out := compute_metric(self, reference, candidate, schema)) is not None:
            return out
        return compute_trend(reference, candidate, schema)
