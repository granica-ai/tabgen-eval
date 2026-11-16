"""Bounded exact local-mass reference computation and shared radius definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

from tabgen_eval._typing import FloatArray, IntArray
from tabgen_eval.execution.runtime.details import ConcentrationDetails, DetailedOutcome, DetailTable
from tabgen_eval.preprocessing.categories import factorize_categories
from tabgen_eval.preprocessing.columns import numeric_values
from tabgen_eval.schema import Schema

if TYPE_CHECKING:
    from tabgen_eval.metrics import Metric


_EPSILON = 1e-8


def pair_indices(n: int, count: int, seed: int) -> tuple[IntArray, IntArray]:
    """Sample distinct position pairs with replacement using a private random stream."""
    rng = np.random.default_rng(seed)
    left = rng.integers(n, size=count, dtype=np.int64)
    right = rng.integers(n - 1, size=count, dtype=np.int64)
    right += right >= left
    return left, right


def radius_labels(metric: Metric) -> list[str]:
    """Return configured radius identifiers in statistic order."""
    return [
        spec.name.removeprefix("mean_") for spec in metric.stats if spec.name.startswith("mean_")
    ]


def fitted_radii(metric: Metric, pair_squared: FloatArray) -> FloatArray:
    """Select real-reference distance quantiles without using candidate observations."""
    distances = np.sqrt(np.asarray(pair_squared, dtype=np.float64))
    quantiles = np.quantile(distances, metric.radius_quantiles) if metric.radius_quantiles else []
    return np.asarray([*metric.fixed_radii, *quantiles], dtype=np.float64)


def squared_thresholds(radii: FloatArray, dtype: DTypeLike) -> FloatArray:
    """Round squared radii once into the selected distance arithmetic precision."""
    with np.errstate(over="ignore"):
        return np.square(np.asarray(radii, dtype=np.float64)).astype(dtype)


def paired_distances(
    numeric: FloatArray, categorical: IntArray, left: IntArray, right: IntArray
) -> FloatArray:
    """Direct selected-precision distances for a bounded sample of position pairs."""
    distances = np.zeros(len(left), dtype=numeric.dtype)
    for column in range(numeric.shape[1]):
        delta = numeric[left, column] - numeric[right, column]
        distances += delta * delta
    categories = np.zeros(len(left), dtype=np.int64)
    for column in range(categorical.shape[1]):
        categories += 2 * (categorical[left, column] != categorical[right, column])
    return distances + categories.astype(numeric.dtype)


def local_counts(
    numeric: FloatArray, categorical: IntArray, radii: FloatArray, *, tile_size: int = 256
) -> IntArray:
    """Count all radii in one bounded symmetric traversal, excluding positional self.

    Only a ``tile_size`` square distance buffer and integer ``[radii, n]``
    output are retained. Numeric sums use explicit column differences, avoiding
    cancellation in squared norms for near-identical records.
    """
    n = len(numeric)
    counts = np.zeros((len(radii), n), dtype=np.int64)
    thresholds = squared_thresholds(radii, numeric.dtype)
    for q in range(0, n, tile_size):
        qe = min(n, q + tile_size)
        for r in range(q, n, tile_size):
            re = min(n, r + tile_size)
            distances = np.zeros((qe - q, re - r), dtype=numeric.dtype)
            for column in range(numeric.shape[1]):
                delta = numeric[q:qe, column, None] - numeric[None, r:re, column]
                distances += delta * delta
            categories = np.zeros((qe - q, re - r), dtype=np.int64)
            for column in range(categorical.shape[1]):
                categories += 2 * (
                    categorical[q:qe, column, None] != categorical[None, r:re, column]
                )
            distances += categories.astype(numeric.dtype)
            for index, threshold in enumerate(thresholds):
                inside = distances <= threshold
                if q == r:
                    np.fill_diagonal(inside, False)
                counts[index, q:qe] += inside.sum(axis=1)
                if q != r:
                    counts[index, r:re] += inside.sum(axis=0)
    return counts


def summarize_counts(counts: IntArray, upper_quantile: float) -> FloatArray:
    """Return mean/upper mass per radius from one already computed count matrix."""
    n = counts.shape[1]
    if n < 2:
        return np.full((len(counts), 2), np.nan)
    return np.column_stack(
        (
            counts.sum(axis=1, dtype=np.int64) / (n * (n - 1)),
            np.quantile(counts, upper_quantile, axis=1) / (n - 1),
        )
    )


@overload
def outcome(
    metric: Metric,
    radii: FloatArray,
    reference: FloatArray,
    candidate: FloatArray,
    counts: tuple[int, int],
    *,
    details: Literal[False] = False,
) -> dict[str, float]: ...


@overload
def outcome(
    metric: Metric,
    radii: FloatArray,
    reference: FloatArray,
    candidate: FloatArray,
    counts: tuple[int, int],
    *,
    details: Literal[True],
) -> DetailedOutcome: ...


@overload
def outcome(
    metric: Metric,
    radii: FloatArray,
    reference: FloatArray,
    candidate: FloatArray,
    counts: tuple[int, int],
    *,
    details: bool,
) -> dict[str, float] | DetailedOutcome: ...


def outcome(
    metric: Metric,
    radii: FloatArray,
    reference: FloatArray,
    candidate: FloatArray,
    counts: tuple[int, int],
    *,
    details: bool = False,
) -> dict[str, float] | DetailedOutcome:
    """Build scalars and optional descriptive tables without numerical reruns."""
    ratios = (candidate + _EPSILON) / (reference + _EPSILON)
    all_values = {
        f"{summary}_{label}": float(ratios[index, column])
        for index, label in enumerate(radius_labels(metric))
        for column, summary in enumerate(("mean", "upper"))
    }
    values = {name: all_values[name] for name in metric.stat_names}
    if not details:
        return values
    labels = np.asarray(radius_labels(metric), dtype=object)
    table = DetailTable(
        {
            "radius_id": labels,
            "radius": np.asarray(radii, dtype=np.float64),
            "reference_mean": reference[:, 0],
            "candidate_mean": candidate[:, 0],
            "mean_ratio": ratios[:, 0],
            "reference_upper": reference[:, 1],
            "candidate_upper": candidate[:, 1],
            "upper_ratio": ratios[:, 1],
            "reference_n": np.full(len(labels), counts[0], dtype=np.int64),
            "candidate_n": np.full(len(labels), counts[1], dtype=np.int64),
        },
        metadata={"upper_quantile": metric.upper_quantile, "epsilon": _EPSILON},
    )
    metadata = {
        "epsilon": _EPSILON,
        "comparison": "(candidate + epsilon) / (reference + epsilon)",
        "upper_quantile": metric.upper_quantile,
        "radius_pairs": metric.radius_pairs if metric.radius_quantiles and counts[0] >= 2 else 0,
        "radius_seed": metric.radius_seed,
        "radius_quantiles": metric.radius_quantiles,
        "fixed_radii": metric.fixed_radii,
        "distance_geometry": "reference_numeric_ranges_and_typed_category_identity",
        "self_exclusion": "position",
        "scope": "point",
    }
    return DetailedOutcome(values, ConcentrationDetails({"radii": table}, metadata))


@overload
def concentration(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    metric: Metric,
    *,
    details: Literal[False] = False,
) -> dict[str, float]: ...


@overload
def concentration(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    metric: Metric,
    *,
    details: Literal[True],
) -> DetailedOutcome: ...


@overload
def concentration(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    metric: Metric,
    *,
    details: bool,
) -> dict[str, float] | DetailedOutcome: ...


def concentration(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    metric: Metric,
    *,
    details: bool = False,
) -> dict[str, float] | DetailedOutcome:
    """Prepare complete rows, fit shared geometry, and evaluate one exact comparison."""
    features = list(schema.numeric) + list(schema.categorical)
    reference, candidate = reference.dropna(subset=features), candidate.dropna(subset=features)
    size = (len(reference), len(candidate))
    n_radii = len(metric.fixed_radii) + len(metric.radius_quantiles)
    if min(size) < 2 or not (schema.numeric or schema.categorical):
        empty = np.full((n_radii, 2), np.nan)
        radii = np.asarray([*metric.fixed_radii, *([np.nan] * len(metric.radius_quantiles))])
        return outcome(metric, radii, empty, empty, size, details=details)
    numbers = [
        np.column_stack([numeric_values(frame[column]) for column in schema.numeric]).astype(
            metric.dtype
        )
        if schema.numeric
        else np.empty((len(frame), 0), dtype=metric.dtype)
        for frame in (reference, candidate)
    ]
    low = numbers[0].min(axis=0)
    ranges = numbers[0].max(axis=0) - low
    ranges[ranges == 0] = 1
    numbers = [np.ascontiguousarray((values - low) / ranges) for values in numbers]
    if not all(np.isfinite(values).all() for values in numbers):
        empty = np.full((n_radii, 2), np.nan)
        return outcome(metric, np.full(n_radii, np.nan), empty, empty, size, details=details)
    codes = [np.empty((n, len(schema.categorical)), dtype=np.int64) for n in size]
    for index, name in enumerate(schema.categorical):
        pair, _ = factorize_categories(reference[name], candidate[name])
        codes[0][:, index], codes[1][:, index] = pair
    if metric.radius_quantiles:
        pair = pair_indices(size[0], metric.radius_pairs, metric.radius_seed)
        distances = paired_distances(numbers[0], codes[0], *pair)
    else:
        distances = np.empty(0)
    radii = fitted_radii(metric, distances)
    profiles = [
        summarize_counts(local_counts(num, cat, radii), metric.upper_quantile)
        for num, cat in zip(numbers, codes, strict=True)
    ]
    return outcome(metric, radii, *profiles, size, details=details)
