"""Owned score distributions with finite summaries and empirical percentile bands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tabgen_eval._typing import is_integer_scalar

from .constants import DEFAULT_CONFIDENCE_INTERVAL


@dataclass(frozen=True)
class ScoreDistribution:
    """One statistic's ordered draws, finite summaries, and raw percentile band.

    ``samples`` is an owned read-only float64 vector. Undefined comparisons
    remain NaN and are counted separately. Summaries use finite draws only;
    every summary is NaN when no draw is finite. Band endpoints are empirical
    linear quantiles at ``(1-confidence_interval)/2`` and ``(1+confidence_interval)/2``, without shifting.
    """

    samples: NDArray[np.float64]
    mean: float
    median: float
    band_low: float
    band_high: float
    finite_count: int
    unavailable_count: int
    confidence_interval: float = DEFAULT_CONFIDENCE_INTERVAL
    discarded_reasons: dict[str, int] = field(default_factory=dict[str, int])

    @property
    def requested_count(self) -> int:
        """Number of requested draws, including nonfinite outcomes."""
        return len(self.samples)

    @property
    def reason(self) -> str | None:
        """Explain why finite summaries are unavailable, when applicable."""
        return "no finite replicate scores" if self.finite_count == 0 else None

    def quantiles(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Return finite-draw quantiles, with the input probability array's shape.

        A scalar probability returns a zero-dimensional array. An unavailable
        distribution returns NaNs with the same shape; neither case changes the
        result's container type.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("quantile probabilities must be finite and between zero and one")
        finite = self.samples[np.isfinite(self.samples)]
        if not finite.size:
            return np.full(probabilities.shape, np.nan)
        return np.asarray(_finite_quantiles(finite, probabilities), dtype=np.float64)

    @classmethod
    def from_samples(
        cls,
        samples: ArrayLike,
        *,
        confidence_interval: float,
        discarded_reasons: Mapping[str, int] | None = None,
    ) -> ScoreDistribution:
        """Own a score vector and summarize its finite outcomes."""
        values = np.array(samples, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError("score samples must be a one-dimensional vector")
        if (
            isinstance(confidence_interval, (bool, np.bool_))
            or not isinstance(confidence_interval, Real)
            or not np.isfinite(confidence_interval)
            or not 0 < confidence_interval < 1
        ):
            raise ValueError("confidence_interval must be a finite number in (0, 1)")
        if discarded_reasons is None:
            reasons = {
                name: int(np.count_nonzero(mask))
                for name, mask in (
                    ("nan", np.isnan(values)),
                    ("positive_infinity", np.isposinf(values)),
                    ("negative_infinity", np.isneginf(values)),
                )
                if np.any(mask)
            }
        else:
            reasons = _validated_reasons(discarded_reasons.items())
            if sum(reasons.values()) != int(np.count_nonzero(~np.isfinite(values))):
                raise ValueError("discarded reason counts must match the nonfinite replicate count")
        values.setflags(write=False)
        finite = values[np.isfinite(values)]
        if len(finite):
            low, high = _finite_quantiles(
                finite, [(1 - confidence_interval) / 2, (1 + confidence_interval) / 2]
            )
            # Scaling keeps finite same-sign draws from overflowing their sum.
            scale = float(np.max(np.abs(finite)))
            average = float((finite / scale).mean() * scale) if scale else 0.0
            median = float(_finite_quantiles(finite, 0.5))
        else:
            average = median = low = high = float("nan")
        return cls(
            values,
            average,
            median,
            float(low),
            float(high),
            len(finite),
            len(values) - len(finite),
            float(confidence_interval),
            reasons,
        )


def _finite_quantiles(values: NDArray[np.float64], probabilities: ArrayLike) -> NDArray[np.float64]:
    """Avoid interpolation overflow for large finite score magnitudes."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scale = float(np.max(np.abs(values)))
    if scale > np.finfo(np.float64).max / 2:
        return np.asarray(
            np.quantile(values / scale, probabilities, method="linear") * scale,
            dtype=np.float64,
        )
    return np.asarray(np.quantile(values, probabilities, method="linear"), dtype=np.float64)


def _validated_reasons(reasons: Iterable[tuple[object, object]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, count in reasons:
        if (
            not isinstance(key, str)
            or not key
            or isinstance(count, (bool, np.bool_))
            or not is_integer_scalar(count)
            or count < 0
        ):
            raise ValueError("discarded_reasons must map nonempty reasons to nonnegative counts")
        if count:
            result[key] = int(count)
    return result
