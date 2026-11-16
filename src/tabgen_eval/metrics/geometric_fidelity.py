"""Local geometric fidelity through precision, recall, density, and coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabgen_eval.execution.python import distance as python_distance
from tabgen_eval.execution.runtime.components import PRDC_COMPONENTS, selected_components
from tabgen_eval.execution.runtime.dispatch import compute_metric

from ..schema import Schema
from .base import Direction, Metric, MetricFamily, StatSpec


class PRDC(Metric):
    """Measure local geometric precision, recall, density, and coverage.

    ``nearest_k=5`` sets within-sample neighborhood radii. Precision is the
    fraction of generated rows inside real neighborhoods; recall is the fraction
    of real rows inside generated neighborhoods. Density counts overlapping real
    neighborhoods, and coverage measures proximity to each real row.
    All are higher-better; density can exceed one. Compare with the real-real
    reference at the same sample size rather than assuming every ideal is one.

    Rows with missing values are excluded. Distances use reference-fitted numeric
    range scaling and full categorical indicators; neighborhood membership uses
    strict distance inequalities. Native execution performs numeric operations
    in the selected precision, so float32 results can differ from float64 at
    neighborhood boundaries.
    Precision, density, and coverage need more than nearest_k usable reference
    rows; recall needs more than nearest_k usable candidate rows. Both sides
    must be nonempty. Shared
    execution options are documented on Metric. Public evaluation uses
    ``nearest_k=5``; internal constructors retain options for backend testing.
    """

    name = "prdc"
    category = "geometric_fidelity"
    family = MetricFamily.SAMPLE_BASED
    symmetric = False
    nearest_k = 5
    requested_components = PRDC_COMPONENTS

    stats = (
        StatSpec("precision", "Precision", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),
        StatSpec("recall", "Recall", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),
        StatSpec("density", "Density", Direction.HIGHER_BETTER, bounds=(0.0, float("inf"))),
        StatSpec("coverage", "Coverage", Direction.HIGHER_BETTER, bounds=(0.0, 1.0)),
    )

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.requested_components = selected_components(self.requested_components, PRDC_COMPONENTS)

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
        return python_distance.prdc(
            reference,
            candidate,
            schema,
            self.nearest_k,
            dtype=self.dtype,
            requested_components=self.requested_components,
        )
