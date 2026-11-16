"""Conditional Shape, Trend, and frequency recovery within reference-defined slices.

The six statistics summarize all retained slices and the rare-slice subset,
with equal weighting within each conditioning column and across columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabgen_eval.execution.python.conditional import (
    any_rare_slice,
    compute_conditional,
)
from tabgen_eval.execution.python.primitives import NUM_BINS
from tabgen_eval.execution.runtime.components import CONDITIONAL_COMPONENTS, selected_components
from tabgen_eval.execution.runtime.dispatch import compute_metric

from ..schema import Schema
from .base import Direction, Metric, MetricFamily, StatSpec


class Conditional(Metric):
    """Measure Shape, Trend, and population recovery within reference-defined slices.

    Reports ``cond_shape_all``, ``cond_shape_rare``, ``cond_trend_all``,
    ``cond_trend_rare``, ``recovery_all``, and ``recovery_rare``. Shape/Trend are
    higher-better similarities in [0, 1]. Recovery compares generated and real
    slice frequencies; one is the target and values can exceed one.

    ``num_bins=10`` partitions numeric conditioning columns into equal-width
    reference-range bins and controls numeric contingency binning. Categorical conditions
    use their observed levels. ``min_count=10`` excludes slices with fewer real
    rows; ``rare_threshold=0.05`` labels eligible slices whose reference fraction
    is strictly below that threshold. Results average eligible slices within
    each conditioning column, then columns, without population weighting.

    Recovery needs one modeled column, Shape two, and conditional Trend three.
    Rare statistics are not applicable when no eligible rare slice exists.
    Shared execution/resource options are documented on Metric. Public
    evaluation uses these fixed computation defaults.
    """

    name = "conditional_fidelity"
    category = "conditional_fidelity"
    family = MetricFamily.SAMPLE_BASED

    rare_threshold = 0.05
    min_count = 10
    num_bins = NUM_BINS  # shared with the contingency binning in primitives.py
    requested_components = CONDITIONAL_COMPONENTS

    stats = (
        StatSpec("cond_shape_all", "Cond Shape", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),
        StatSpec(
            "cond_shape_rare", "Cond Shape (Rare)", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)
        ),
        StatSpec("cond_trend_all", "Cond Trend", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),
        StatSpec(
            "cond_trend_rare", "Cond Trend (Rare)", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)
        ),
        StatSpec(
            "recovery_all", "Recovery", Direction.TARGET, target=1.0, bounds=(0.0, float("inf"))
        ),
        StatSpec(
            "recovery_rare",
            "Recovery (Rare)",
            Direction.TARGET,
            target=1.0,
            bounds=(0.0, float("inf")),
        ),
    )

    # -- applicability ---------------------------------------------------

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.requested_components = selected_components(
            self.requested_components, CONDITIONAL_COMPONENTS
        )

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        na: set[str] = set()
        cols = list(schema.numeric) + list(schema.categorical)
        if not cols:
            return set(self.stat_names)
        if len(cols) < 2:
            na |= {"cond_shape_all", "cond_shape_rare"}
        if len(cols) < 3:
            # Conditional Trend needs a *pair* of other columns within a slice (≥ 3 total).
            na |= {"cond_trend_all", "cond_trend_rare"}
        if reference is None:
            raise ValueError("Conditional applicability requires reference observations")
        if not self._any_rare(reference, schema):
            na |= {"cond_shape_rare", "cond_trend_rare", "recovery_rare"}
        return na

    def _any_rare(self, real: pd.DataFrame, schema: Schema) -> bool:
        return any_rare_slice(
            real,
            schema,
            num_bins=self.num_bins,
            min_count=self.min_count,
            rare_threshold=self.rare_threshold,
        )

    # -- computation -----------------------------------------------------

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
        return compute_conditional(
            reference,
            candidate,
            schema,
            num_bins=self.num_bins,
            min_count=self.min_count,
            rare_threshold=self.rare_threshold,
            requested_components=self.requested_components,
        )
