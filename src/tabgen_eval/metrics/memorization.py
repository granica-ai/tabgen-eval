"""Exact nearest-record distances against the generator training data."""

from __future__ import annotations

from typing import TypeGuard

import numpy as np
import pandas as pd

from tabgen_eval.execution.runtime.outcomes import DCR_PERCENTILES
from tabgen_eval.schema import Schema

from .base import Direction, Metric, StatSpec


class _MemorizationMetric(Metric):
    """Shared optional row budgets and numerical dispatch for point metrics."""

    category = "memorization"
    bootstrapped = False
    max_samples: int | None = None

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        from tabgen_eval.execution.runtime.memorization_sampling import sample_limit

        self.max_samples = sample_limit(self.max_samples)

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        return set(self.stat_names) if not schema.numeric and not schema.categorical else set()

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> dict[str, float]:
        """Compute point scores with generator training data as ``reference``."""
        from tabgen_eval.execution.runtime.memorization import compute_point

        return compute_point(self, reference, candidate, schema, rng=rng)


class DCR(_MemorizationMetric):
    """Generated-to-training nearest distances and their training baseline ratio.

    ``q05``, ``q10``, ``q25``, ``q50``, ``q75``, ``q90``, ``q95`` and ``q99``
    summarize generated-to-training Euclidean distances. Matching ``holdout_qXX``
    statistics summarize holdout-to-training distances. ``median`` aliases
    generated ``q50``. ``ratio`` divides
    that median by the independent real holdout's median nearest-training distance.
    Both queries use the same selected training pool and fitted geometry. Records
    with identical values across pools have zero distance. Missing or unusable
    holdout data and a zero denominator leave the ratio undefined, while raw
    generated distances remain available. Quantiles interpolate distances linearly.

    Numeric ranges and categorical geometry are fitted on selected training
    rows. By default, ``max_samples=None`` retains every row. A positive integer
    independently caps each source pool using uniform sampling without
    replacement. Smaller pools retain every row and are not volume matched.

    ``max_holdout_samples=None`` separately retains the full independent holdout;
    a positive integer caps only that pool. Its draw does not alter training or
    generated selections. ``compute(..., holdout=...)`` accepts this third role.

    Search is exact on all supported backends.

    Public ``evaluate(..., metrics=["dcr_ratio"])`` requires train, test, and
    generated data. Optional holdout behavior here belongs to the lower-level
    distance adapter, which can also provide raw distances without a ratio.
    """

    name = "dcr"
    search_method = "exact"
    max_holdout_samples: int | None = None

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        from tabgen_eval.execution.runtime.memorization_sampling import sample_limit

        self.max_holdout_samples = sample_limit(self.max_holdout_samples)
        if self.search_method != "exact":
            raise ValueError("DCR only supports search_method='exact'")

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
        holdout: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute generated distances and an optional independent real-holdout ratio."""
        from tabgen_eval.execution.runtime.memorization import compute_point

        return compute_point(self, reference, candidate, schema, rng=rng, holdout=holdout)

    stats = (
        StatSpec("median", "DCR (Median)", Direction.LOWER_BETTER, bounds=(0.0, float("inf"))),
        *(
            StatSpec(
                f"q{percentile:02d}",
                "DCR (5th percentile)" if percentile == 5 else f"DCR (Generated q{percentile:02d})",
                Direction.LOWER_BETTER,
                bounds=(0.0, float("inf")),
            )
            for percentile in DCR_PERCENTILES
        ),
        *(
            StatSpec(
                f"holdout_q{percentile:02d}",
                f"DCR (Holdout q{percentile:02d})",
                Direction.LOWER_BETTER,
                bounds=(0.0, float("inf")),
            )
            for percentile in DCR_PERCENTILES
        ),
        StatSpec(
            "ratio",
            "DCR Ratio",
            Direction.TARGET,
            target=1.0,
            bounds=(0.0, float("inf")),
            compare_to_ideal=False,
        ),
    )


class NearMatchRate(_MemorizationMetric):
    """Fraction of generated records lying near at least one training record.

    For each generated row, find whether any selected training row lies within
    the fixed threshold ``1e-3``, inclusively. Divide the number of
    matching generated rows by the retained generated-row count. Repeated copies
    of one training record each contribute, so this measures how much generated
    output reproduces the training set, not how much of training was reproduced.

    Distances use the training-fitted normalized numeric and one-hot categorical
    geometry. The default ``max_samples=None`` retains all rows; a positive integer
    independently caps each complete-case pool. This proximity diagnostic is
    neither exact record equality nor a privacy guarantee.
    """

    name = "near_match_rate"
    stats = (StatSpec("rate", "Generated near-match rate", Direction.LOWER_BETTER),)


class ReplicationRatio(_MemorizationMetric):
    """Fraction of training rows reached by at least one generated row.

    Restores training coverage under the name replication ratio. A retained
    training row counts if its nearest generated row is within
    the fixed inclusive threshold ``1e-3``. Divide by the retained training
    row count. Repeated generated copies do not increase coverage; duplicate
    training rows each count separately. Geometry is fitted on selected training
    rows, using the same numeric scaling and categorical encoding as DCR.

    This point-only metric requires the generator's actual training data and no
    holdout. ``max_samples=None`` retains all rows; a positive integer caps each
    pool independently. With usable training but no usable generated rows the
    ratio is zero; with no usable training rows it is undefined.
    """

    name = "replication_ratio"
    stats = (StatSpec("rate", "Training replication ratio", Direction.LOWER_BETTER),)


def uses_memorization_protocol(metric: Metric) -> TypeGuard[_MemorizationMetric]:
    """Whether a metric uses fixed training-referenced row selection and execution."""
    return isinstance(metric, _MemorizationMetric)
