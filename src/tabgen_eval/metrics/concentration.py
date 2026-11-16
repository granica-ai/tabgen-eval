"""Local neighborhood mass compared with an independent real reference."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from tabgen_eval.execution.runtime.dispatch import compute_metric
from tabgen_eval.schema import Schema

from .base import Direction, Metric, MetricFamily, StatSpec


def concentration_specs(
    fixed_radii: tuple[float, ...], radius_quantiles: tuple[float, ...]
) -> tuple[StatSpec, ...]:
    """Identify fixed radii by value and fitted radii by reference quantile."""
    labels = [
        "radii_"
        + (
            "0"
            if radius == 0
            else np.format_float_scientific(
                radius,
                unique=True,
                trim="-",
                exp_digits=1,
            )
        )
        for radius in fixed_radii
    ]
    labels += [
        (f"q{int(100 * q):02d}" if (100 * q).is_integer() else f"q{100 * q:.15g}").replace(".", "_")
        for q in radius_quantiles
    ]
    return tuple(
        StatSpec(
            f"{summary}_{label}",
            f"{display} concentration ratio ({label})",
            Direction.TARGET,
            bounds=(0.0, float("inf")),
            target=1.0,
        )
        for label in labels
        for summary, display in (("mean", "Mean"), ("upper", "Upper-quantile"))
        if (summary, label) != ("upper", "radii_1e-3")
    )


class Concentration(Metric):
    """Compare within-dataset neighborhood mass at shared real-defined radii.

    For every row, count other row positions within inclusive Euclidean radius
    ``r`` and divide by ``n-1``. Repeated records remain neighbors. At each radius,
    report smoothed candidate/reference ratios of mean and upper-quantile mass:
    ``(candidate + epsilon) / (reference + epsilon)``. Ratios above one indicate
    more concentration; equal profiles, including two zero masses, give one.
    A fixed ``epsilon=1e-8`` stabilizes small denominators. Both directions are
    compared with the real-real reference, rather than treating greater concentration as better.

    Defaults include radius ``1e-3`` and the 5th/25th percentiles of real-reference
    pair distances. ``radius_pairs=16384`` bounds the independently seeded sample
    used to select radii. ``upper_quantile=0.95`` selects the mass quantile.
    ``fixed_radii`` and ``radius_quantiles`` are configurable nonempty combined
    radius specifications; fixed radii use the reference-fitted distance units.
    Fixed-radius keys include the value, e.g. ``mean_radii_1e-3``; the
    fixed radius ``1e-3`` reports only the mean. Other radii also report
    upper-quantile keys, e.g. ``upper_q05``.

    Numeric ranges are fitted on reference rows. Categorical distances retain
    distinct observed identities, including generated-only levels. Records missing
    any declared distance feature are excluded before sampling or uncertainty
    draws; source exclusion counts remain visible in ``report.row_usage()``. Fewer than two
    usable rows on either side produce undefined statistics. Detailed output
    contains both raw concentration profiles, actual radii and retained counts.
    Unfiltered evaluation includes this metric when its data roles are supplied.
    Distance searches are exact and their work is quadratic in row count.
    """

    name = "concentration"
    category = "concentration"
    family = MetricFamily.SAMPLE_BASED
    fixed_radii = (1e-3,)
    radius_quantiles = (0.05, 0.25)
    upper_quantile = 0.95
    radius_pairs = 16384
    radius_seed = 1729
    stats = concentration_specs(fixed_radii, radius_quantiles)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        for name in ("fixed_radii", "radius_quantiles"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite numbers")
            if (values < 0).any() or (name == "radius_quantiles" and (values > 1).any()):
                raise ValueError(f"Invalid {name}")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"{name} cannot contain duplicates")
            setattr(self, name, tuple(values.tolist()))
        if not self.fixed_radii and not self.radius_quantiles:
            raise ValueError("At least one concentration radius is required")
        if not np.isfinite(self.upper_quantile) or not 0 <= self.upper_quantile <= 1:
            raise ValueError("upper_quantile must lie in [0, 1]")
        for name, minimum in (("radius_pairs", 1), ("radius_seed", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.stats = concentration_specs(self.fixed_radii, self.radius_quantiles)

    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator | None,
    ) -> dict[str, float]:
        """Return concentration ratios using the selected execution backend."""
        if (result := compute_metric(self, reference, candidate, schema)) is not None:
            return result
        from tabgen_eval.execution.python.concentration import concentration

        return concentration(reference, candidate, schema, self)

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        """A distance profile requires at least one declared feature."""
        return set() if schema.numeric or schema.categorical else set(self.stat_names)
