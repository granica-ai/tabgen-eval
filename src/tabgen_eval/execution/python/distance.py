"""Python PRDC and DCR computation over reference-fitted distance geometry.

Metric adapters select execution and supply explicit parameters. This module
owns complete-row filtering, feature encoding and distance reductions only;
sampling, calibration pool assignment and report assembly remain with callers.
Dense and compact category representations share strict squared-distance kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

from tabgen_eval._typing import FloatArray
from tabgen_eval.execution.runtime.components import PRDC_COMPONENTS, selected_components
from tabgen_eval.execution.runtime.outcomes import DCR_PERCENTILES, MemorizationOutcome
from tabgen_eval.schema import Schema

from .encoding.distances import (
    NearestRecordIndex,
    distance_tiles,
    kth_neighbor_squared,
    prepare_features,
    refine_distances,
)
from .encoding.features import StructuredFeatures, fit_encoder

if TYPE_CHECKING:
    from tabgen_eval.metrics import Metric


NAN = float("nan")
_STATS = PRDC_COMPONENTS

# Encoded widths above this threshold use compact category codes.
_STRUCTURED_MIN_DIM = 512


def _compute_prdc(
    real: FloatArray | StructuredFeatures,
    fake: FloatArray | StructuredFeatures,
    k: int,
    *,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float]:
    """Reduce PRDC's strict manifold membership using shared squared distances."""
    real, fake = prepare_features(real), prepare_features(fake)
    n_real, n_fake = real.n_rows, fake.n_rows
    components = selected_components(requested_components, PRDC_COMPONENTS)
    needs_real_radius = any(name != "recall" for name in components)
    needs_fake_radius = "recall" in components
    wants_counts = "precision" in components or "density" in components
    wants_coverage = "coverage" in components
    if (
        not n_real
        or not n_fake
        or (needs_real_radius and n_real <= k)
        or (needs_fake_radius and n_fake <= k)
        or not np.isfinite(real.numeric).all()
        or not np.isfinite(fake.numeric).all()
    ):
        return {stat: NAN for stat in _STATS}
    if not components:
        return {stat: NAN for stat in _STATS}
    real_radius = kth_neighbor_squared(real, k) if needs_real_radius else None
    fake_radius = kth_neighbor_squared(fake, k) if needs_fake_radius else None
    real_balls_containing_fake = np.zeros(n_fake, dtype=np.int64) if wants_counts else None
    real_inside_any_fake = np.zeros(n_real, dtype=bool) if needs_fake_radius else None
    nearest_fake = np.full(n_real, np.inf) if wants_coverage else None
    for real_rows, fake_rows, distances in distance_tiles(real, fake):
        refine_distances(
            real.rows(real_rows.start, real_rows.stop),
            fake.rows(fake_rows.start, fake_rows.stop),
            distances,
            query_thresholds=None if real_radius is None else real_radius[real_rows],
            reference_thresholds=None if fake_radius is None else fake_radius[fake_rows],
        )
        if wants_coverage:
            nearest_fake[real_rows] = np.minimum(nearest_fake[real_rows], distances.min(axis=1))
        if wants_counts:
            real_balls_containing_fake[fake_rows] += (distances < real_radius[real_rows, None]).sum(
                axis=0
            )
        if needs_fake_radius:
            real_inside_any_fake[real_rows] |= (distances < fake_radius[None, fake_rows]).any(
                axis=1
            )
    return {
        "precision": float((real_balls_containing_fake > 0).mean())
        if "precision" in components
        else NAN,
        "recall": float(real_inside_any_fake.mean()) if needs_fake_radius else NAN,
        "density": float(real_balls_containing_fake.mean() / float(k))
        if "density" in components
        else NAN,
        "coverage": float((nearest_fake < real_radius).mean()) if wants_coverage else NAN,
    }


def compute_prdc(
    real: FloatArray,
    fake: FloatArray,
    k: int,
    *,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float]:
    """PRDC on dense coordinates with float64 squared-distance arithmetic.

    Supplied encoded coordinate values are preserved. Radii and cross-distances
    use the same shared kernel, so tied boundaries retain strict ``<`` semantics
    without square-root rounding.
    """
    return _compute_prdc(real, fake, k, requested_components=requested_components)


def compute_prdc_structured(
    real: StructuredFeatures,
    fake: StructuredFeatures,
    k: int,
    *,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float]:
    """PRDC on compact category codes, using the same float64 distance kernels.

    Numeric coordinates retain the dense path's encoded values. Unknown levels
    preserve zero-one-hot semantics without allocating category-width matrices.
    """
    return _compute_prdc(real, fake, k, requested_components=requested_components)


def prdc(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    nearest_k: int,
    *,
    dtype: DTypeLike = np.float64,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float]:
    """Compute PRDC with complete-row filtering and reference-fitted encoding.

    Row order and duplicate multiplicities are preserved. Unknown candidate
    categories have all-zero one-hot representations. Encoded widths above
    _STRUCTURED_MIN_DIM use compact category codes; smaller designs use dense
    coordinates. Both paths scale in float64 and cast coordinates to *dtype*
    (float64 by default), with float64 squared-distance arithmetic. Inputs are
    read-only; no source sampling occurs.
    """
    # Drop NaN rows before building features (distance metrics need complete rows).
    ref = reference.dropna(subset=list(schema.numeric) + list(schema.categorical))
    cand = candidate.dropna(subset=list(schema.numeric) + list(schema.categorical))
    if not len(ref) or not len(cand):
        return {s: NAN for s in _STATS}
    enc = fit_encoder(ref, schema)
    if enc.onehot_dim > _STRUCTURED_MIN_DIM:
        # Wide one-hot space: compute distances from the structured representation, which
        # gives the same PRDC without ever building the one-hot columns.
        return compute_prdc_structured(
            enc.transform_structured(ref, dtype=dtype),
            enc.transform_structured(cand, dtype=dtype),
            nearest_k,
            requested_components=requested_components,
        )
    return compute_prdc(
        enc.transform(ref, dtype=dtype),
        enc.transform(cand, dtype=dtype),
        nearest_k,
        requested_components=requested_components,
    )


def compute_memorization_group(
    training: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    metrics: Sequence[Metric],
    *,
    holdout: pd.DataFrame | None = None,
) -> dict[str, MemorizationOutcome]:
    """Share training-fitted exact searches for memorization metrics."""
    from tabgen_eval.execution.runtime.distance_outcomes import (
        NEAR_MATCH_THRESHOLD,
        memorization_metric_outcome,
    )

    if any(
        metric.name not in {"dcr", "near_match_rate", "replication_ratio"} for metric in metrics
    ):
        raise ValueError(
            "Only DCR, near_match_rate and replication_ratio are supported by memorization execution"
        )
    train, gen = (
        training.dropna(subset=list(schema.numeric) + list(schema.categorical)),
        generated.dropna(subset=list(schema.numeric) + list(schema.categorical)),
    )
    held = (
        None
        if holdout is None
        else holdout.dropna(subset=list(schema.numeric) + list(schema.categorical))
    )
    median = q05 = holdout_median = np.nan
    forward = reverse = None
    wants_replication = any(metric.name == "replication_ratio" for metric in metrics)
    wants_forward = any(metric.name in {"dcr", "near_match_rate"} for metric in metrics)
    wants_dcr = any(metric.name == "dcr" for metric in metrics)
    generated_quantiles, holdout_quantiles = {}, {}
    if len(train):
        encoder = fit_encoder(train, schema)
        encoded_train = encoder.transform_structured(train, dtype=np.float64)
        index = NearestRecordIndex(encoded_train) if wants_forward else None
        if len(gen):
            encoded_gen = encoder.transform_structured(gen, dtype=np.float64)
            if wants_forward:
                forward = index.query(encoded_gen)
            if wants_replication:
                reverse = NearestRecordIndex(encoded_gen).query(encoded_train)
        if wants_dcr and forward is not None:
            distances = forward
            median = float(np.median(distances))
            generated_quantiles = dict(
                zip(
                    DCR_PERCENTILES,
                    np.quantile(distances, np.asarray(DCR_PERCENTILES) / 100, method="linear"),
                    strict=True,
                )
            )
            generated_quantiles[50] = median
            q05 = generated_quantiles[5]
        if wants_dcr and held is not None and len(held):
            distances = index.query(encoder.transform_structured(held, dtype=np.float64))
            holdout_quantiles = dict(
                zip(
                    DCR_PERCENTILES,
                    np.quantile(distances, np.asarray(DCR_PERCENTILES) / 100, method="linear"),
                    strict=True,
                )
            )
            holdout_median = float(np.median(distances))
            holdout_quantiles[50] = holdout_median

    def rate(metric: Metric) -> float:
        distances = forward if metric.name == "near_match_rate" else reverse
        if metric.name == "replication_ratio" and len(train) and not len(gen):
            return 0.0
        if distances is None or np.isnan(distances).any():
            return np.nan
        return float((distances <= NEAR_MATCH_THRESHOLD).mean())

    return {
        metric.name: memorization_metric_outcome(
            metric,
            median=median,
            q05=q05,
            holdout_median=holdout_median,
            generated_quantiles=generated_quantiles,
            holdout_quantiles=holdout_quantiles,
            holdout_count=None if held is None else len(held),
            counts=(len(train), len(gen)),
            rate=rate(metric) if metric.name != "dcr" else np.nan,
            diagnostics={"search_method": "exact"},
        )
        for metric in metrics
    }
