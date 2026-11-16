"""Shared exact distance operations for PRDC and DCR.

Encoding owns the reference-fitted geometry. These kernels promote its existing
coordinates to float64, keep distances squared until a metric needs Euclidean
units, and preserve the zero-vector meaning of unknown categorical levels.
Float64 matrix products handle ordinary distances; direct differences resolve
cancellation and uncertain rankings or strict boundary comparisons.
Tiles bound pairwise storage independently of the number of query/reference rows.
No samples, encoders, or computed distances are shared across metric evaluations.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from sklearn import config_context
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from tabgen_eval._typing import BoolArray, FloatArray, IntArray

from .features import StructuredFeatures

_TILE_BYTES = 8 * 1024**2
_MAX_QUERY_ROWS = 1024
_KD_TREE_MAX_DIM = 5
_DIRECT_MAX_DIM = 15
_ROUNDING_MARGIN = 1e-10


def prepare_features(features: NDArray[np.generic] | StructuredFeatures) -> StructuredFeatures:
    """Promote numeric coordinates once, preserving dense or compact geometry.

    Dense one-hot arrays become a numeric-only representation. Compact category
    codes and unknown masks are retained without expanding one-hot columns.
    Float64 conversion changes arithmetic precision, not the encoded values.
    """
    if isinstance(features, StructuredFeatures):
        numeric = np.asarray(features.numeric, dtype=np.float64)
        if numeric.ndim != 2 or features.codes.shape != features.oov.shape:
            raise ValueError("distance features require aligned two-dimensional blocks")
        if features.codes.ndim != 2 or features.codes.shape[0] != numeric.shape[0]:
            raise ValueError("numeric/category feature blocks must have equal row counts")
        return StructuredFeatures(numeric, features.codes, features.oov)
    numeric = np.asarray(features, dtype=np.float64)
    if numeric.ndim != 2:
        raise ValueError("dense distance features must be two-dimensional")
    codes = np.empty((len(numeric), 0), dtype=np.int64)
    return StructuredFeatures(numeric, codes, codes < 0)


def _check_layout(query: StructuredFeatures, reference: StructuredFeatures) -> None:
    """Reject incompatible representations before computing any distances."""
    if (
        query.numeric.shape[1] != reference.numeric.shape[1]
        or query.codes.shape[1] != reference.codes.shape[1]
    ):
        raise ValueError("reference/query feature layouts must match")


def _binary(features: StructuredFeatures) -> bool:
    """Binary coordinates have exact integer squared distances in this geometry."""
    return bool(((features.numeric == 0) | (features.numeric == 1)).all())


def _uses_gram(query: StructuredFeatures, reference: StructuredFeatures) -> bool:
    """Choose norm/dot arithmetic at the established numeric-width crossover."""
    return query.numeric.shape[1] > _DIRECT_MAX_DIM


def _needs_refinement(query: StructuredFeatures, reference: StructuredFeatures) -> bool:
    """Identify Gram arithmetic that lacks exact binary-coordinate distances."""
    return _uses_gram(query, reference) and not (_binary(query) and _binary(reference))


def _tolerance(query: StructuredFeatures, reference: StructuredFeatures) -> FloatArray:
    """Per-query margin using the largest reference norm, avoiding a full matrix."""
    qnorm = np.einsum("ij,ij->i", query.numeric, query.numeric)
    rnorm = np.einsum("ij,ij->i", reference.numeric, reference.numeric)
    return (_ROUNDING_MARGIN * (1.0 + qnorm + rnorm.max()))[:, None]


def _paired_squared(
    query: StructuredFeatures,
    reference: StructuredFeatures,
    query_rows: IntArray,
    reference_rows: IntArray,
    *,
    categories: bool = True,
) -> FloatArray:
    """Direct distances for selected pairs, without a pairs-by-features allocation."""
    result = np.zeros(len(query_rows))
    for column in range(query.numeric.shape[1]):
        delta = query.numeric[query_rows, column] - reference.numeric[reference_rows, column]
        result += delta * delta
    if categories:
        for column in range(query.codes.shape[1]):
            different = query.codes[query_rows, column] != reference.codes[reference_rows, column]
            one_unknown = query.oov[query_rows, column] != reference.oov[reference_rows, column]
            result += different
            result += different
            result -= one_unknown
    return result


def _categorical_distances(
    query: StructuredFeatures, reference: StructuredFeatures, result: FloatArray
) -> None:
    """Add full-one-hot category contributions in place."""
    for column in range(query.codes.shape[1]):
        different = query.codes[:, column, None] != reference.codes[None, :, column]
        one_unknown = query.oov[:, column, None] != reference.oov[None, :, column]
        result += different
        result += different
        result -= one_unknown


def _replace_pairs(
    query: StructuredFeatures,
    reference: StructuredFeatures,
    result: FloatArray,
    mask: BoolArray,
    *,
    categories: bool = True,
) -> None:
    """Repair selected pairs; dense uncertainty is cheaper as a direct whole tile."""
    count = np.count_nonzero(mask)
    if not count:
        return
    if count > result.size // 8:
        if query.numeric.shape[1]:
            result[:] = pairwise_distances(query.numeric, reference.numeric, metric="sqeuclidean")
        else:
            result.fill(0.0)
        if categories:
            _categorical_distances(query, reference, result)
    else:
        rows, columns = np.nonzero(mask)
        result[rows, columns] = _paired_squared(
            query, reference, rows, columns, categories=categories
        )


def squared_distances(
    query: NDArray[np.generic] | StructuredFeatures,
    reference: NDArray[np.generic] | StructuredFeatures,
) -> FloatArray:
    """Float64 squared Euclidean distances for one caller-sized block.

    Wide numeric blocks use sklearn's matrix-product path. Small numeric blocks
    use direct differences, which are cheaper at low dimension. Near-zero Gram
    results are repaired directly to preserve duplicates and avoid cancellation.
    Remaining rounding is resolved by ``refine_distances`` before rank/boundary
    reductions. Binary and compact category contributions are exact integers:
    0 for equal levels, 2 for different known levels, 1 for exactly one unknown.
    Call ``distance_tiles`` for bounded-memory processing of large pools.
    """
    query, reference = prepare_features(query), prepare_features(reference)
    _check_layout(query, reference)
    if not query.n_rows or not reference.n_rows or not query.numeric.shape[1]:
        result = np.zeros((query.n_rows, reference.n_rows))
    elif _uses_gram(query, reference):
        result = pairwise_distances(
            query.numeric, reference.numeric, metric="euclidean", squared=True
        )
        if _needs_refinement(query, reference):
            _replace_pairs(
                query, reference, result, result <= _tolerance(query, reference), categories=False
            )
    else:
        result = pairwise_distances(query.numeric, reference.numeric, metric="sqeuclidean")
    _categorical_distances(query, reference, result)
    return result


def refine_distances(
    query: NDArray[np.generic] | StructuredFeatures,
    reference: NDArray[np.generic] | StructuredFeatures,
    result: FloatArray,
    *,
    query_thresholds: FloatArray | None = None,
    reference_thresholds: FloatArray | None = None,
) -> None:
    """Resolve distances near rank or strict-membership thresholds in place.

    This refines the compared values rather than replacing strict ``<`` with a
    tolerance. Ordinary well-separated pairs keep their fast float64 distance.
    Direct low-dimensional kernels and binary geometry need no repair.
    """
    query, reference = prepare_features(query), prepare_features(reference)
    if not _needs_refinement(query, reference):
        return
    tolerance = _tolerance(query, reference)
    mask = np.zeros(result.shape, dtype=bool)
    difference = np.empty_like(result)
    if query_thresholds is not None:
        np.subtract(result, query_thresholds[:, None], out=difference)
        np.abs(difference, out=difference)
        np.less_equal(difference, tolerance, out=mask)
    if reference_thresholds is not None:
        np.subtract(result, reference_thresholds[None, :], out=difference)
        np.abs(difference, out=difference)
        mask |= difference <= tolerance
    _replace_pairs(query, reference, result, mask)


def distance_tiles(
    query: NDArray[np.generic] | StructuredFeatures,
    reference: NDArray[np.generic] | StructuredFeatures,
) -> Iterator[tuple[slice, slice, FloatArray]]:
    """Yield query/reference slices and squared-distance tiles of at most eight MiB.

    Both axes are tiled, so even one very large reference does not force an
    oversized distance row. Numeric input buffers and metric reduction buffers
    are separate from this tile budget. Empty pools yield no tiles.
    """
    query, reference = prepare_features(query), prepare_features(reference)
    _check_layout(query, reference)
    if not query.n_rows or not reference.n_rows:
        return
    elements = max(1, _TILE_BYTES // np.dtype(np.float64).itemsize)
    query_rows = min(_MAX_QUERY_ROWS, query.n_rows, elements)
    reference_rows = min(reference.n_rows, max(1, elements // query_rows))
    for start in range(0, query.n_rows, query_rows):
        end = min(start + query_rows, query.n_rows)
        for ref_start in range(0, reference.n_rows, reference_rows):
            ref_end = min(ref_start + reference_rows, reference.n_rows)
            yield (
                slice(start, end),
                slice(ref_start, ref_end),
                squared_distances(query.rows(start, end), reference.rows(ref_start, ref_end)),
            )


def kth_neighbor_squared(features: NDArray[np.generic] | StructuredFeatures, k: int) -> FloatArray:
    """Squared k-th-neighbor radii from batched sklearn searches.

    Low dimensions retain KD-trees; other dense spaces retain accelerated
    float64 search. Direct distances give the final squared radii. One extra
    candidate checks continuous-space rank separation; only uncertain rows
    use a direct library query. Binary spaces need just k+1 candidates.
    Self occupies rank zero; genuine duplicates retain their multiplicities.
    Too few rows or nonfinite coordinates produce NaN radii.
    """
    if k < 1:
        raise ValueError("neighbor rank k must be positive")
    features = prepare_features(features)
    result = np.full(features.n_rows, np.nan)
    if features.n_rows <= k or not np.isfinite(features.numeric).all():
        return result
    dimension = features.numeric.shape[1]
    use_tree = dimension <= _KD_TREE_MAX_DIM
    binary = _binary(features)
    candidates = min(k + 1 + int(not use_tree and not binary), features.n_rows)
    if features.codes.shape[1] or not dimension or candidates * 16 > _TILE_BYTES:
        return _tiled_kth_neighbor_squared(features, features, k)
    metric = "euclidean" if use_tree else "sqeuclidean"
    model = NearestNeighbors(
        n_neighbors=candidates,
        metric=metric,
        algorithm="kd_tree" if use_tree else "brute",
        n_jobs=1,
    ).fit(features.numeric)
    batch_rows = min(_MAX_QUERY_ROWS, max(1, _TILE_BYTES // (candidates * 16)))
    norms = np.einsum("ij,ij->i", features.numeric, features.numeric)
    max_norm = norms.max()
    # Preserve accelerated search on ordinary input; exceptionally huge source
    # pools must use two-axis tiles to bound even one reference-distance row.
    if features.n_rows * 8 > _TILE_BYTES:
        return _tiled_kth_neighbor_squared(features, features, k)
    with config_context(working_memory=_TILE_BYTES / 1024**2):
        for start in range(0, features.n_rows, batch_rows):
            end = min(start + batch_rows, features.n_rows)
            approximate, neighbors = model.kneighbors(features.numeric[start:end])
            rows = np.arange(start, end)
            if use_tree or binary:
                result[start:end] = _paired_squared(features, features, rows, neighbors[:, k])
                continue
            # Well-separated ranks need only the selected neighbor's direct
            # distance. Recompute other candidates only for uncertain rows.
            radius = _paired_squared(features, features, rows, neighbors[:, k])
            squared = approximate if metric == "sqeuclidean" else approximate**2
            tolerance = _ROUNDING_MARGIN * (1.0 + norms[rows] + max_norm)
            separated = squared[:, k - 1] + tolerance < radius
            if candidates < features.n_rows:
                separated &= squared[:, -1] - tolerance > radius
            uncertain = np.flatnonzero(~separated)
            if uncertain.size:
                selected = neighbors[uncertain]
                exact = _paired_squared(
                    features, features, np.repeat(rows[uncertain], candidates), selected.ravel()
                ).reshape(len(uncertain), candidates)
                repaired = np.partition(exact, k, axis=1)[:, k]
                # k+1 exact zero pairs certify a zero radius. Other ranks are
                # certain if all omitted candidates lie beyond this margin.
                if candidates < features.n_rows:
                    retry = np.flatnonzero(
                        (repaired > 0) & (squared[uncertain, -1] - repaired <= tolerance[uncertain])
                    )
                    if retry.size:
                        # Resolve remaining uncertainty in one bounded query
                        # using sklearn's direct squared-distance backend.
                        with config_context(enable_cython_pairwise_dist=False):
                            direct, _ = model.kneighbors(
                                features.numeric[rows[uncertain[retry]]], n_neighbors=k + 1
                            )
                        repaired[retry] = direct[:, -1]
                radius[uncertain] = repaired
            result[start:end] = radius
    return result


def _tiled_kth_neighbor_squared(
    query: StructuredFeatures, reference: StructuredFeatures, k: int
) -> FloatArray:
    """Chunked rank selection for compact features or oversized dense searches."""
    result = np.full(query.n_rows, np.nan)
    previous = None
    best = None
    for query_rows, reference_rows, distances in distance_tiles(query, reference):
        if previous != query_rows:
            if previous is not None:
                result[previous] = best.max(axis=1)
            previous = query_rows
            best = np.full((query_rows.stop - query_rows.start, k + 1), np.inf)
        # Direct differences give exact zero for each query's own source row;
        # equal records at other positions remain separate ranking weights.
        keep = min(k + 1, distances.shape[1])
        queries = query.rows(query_rows.start, query_rows.stop)
        references = reference.rows(reference_rows.start, reference_rows.stop)
        if _needs_refinement(queries, references):
            threshold = np.partition(distances, keep - 1, axis=1)[:, keep - 1].copy()
            # Every retained candidate must be direct: a later reference tile
            # can place the global rank anywhere inside this tile's top k+1.
            mask = distances <= threshold[:, None] + _tolerance(queries, references)
            _replace_pairs(queries, references, distances, mask)
        distances.partition(keep - 1, axis=1)
        candidates = np.concatenate([best, distances[:, :keep]], axis=1)
        candidates.partition(k, axis=1)
        best = candidates[:, : k + 1].copy()
    result[previous] = best.max(axis=1)
    return result


class NearestRecordIndex:
    """Exact nearest-record queries using the shared float64 distance geometry.

    A reference index can serve several queries within one metric draw. Small
    numeric-only spaces use SciPy's exact KD-tree; other spaces reduce shared
    squared-distance tiles. Query results are Euclidean distances, ready for
    DCR's empirical quantile reductions.
    """

    def __init__(self, reference: NDArray[np.generic] | StructuredFeatures) -> None:
        """Retain promoted geometry and optionally build an exact numeric KD-tree."""
        self.reference = prepare_features(reference)
        self.valid = bool(self.reference.n_rows and np.isfinite(self.reference.numeric).all())
        self.tree = (
            cKDTree(self.reference.numeric)
            if self.valid
            and not self.reference.codes.shape[1]
            and 0 < self.reference.numeric.shape[1] <= 15
            else None
        )

    def query(self, query: NDArray[np.generic] | StructuredFeatures) -> FloatArray:
        """Nearest Euclidean distance per row; invalid geometry gives NaNs."""
        query = prepare_features(query)
        _check_layout(query, self.reference)
        if not self.valid or not np.isfinite(query.numeric).all():
            return np.full(query.n_rows, np.nan)
        if self.tree is not None:
            return self.tree.query(query.numeric, k=1, workers=1)[0]
        nearest = np.full(query.n_rows, np.inf)
        for query_rows, reference_rows, distances in distance_tiles(query, self.reference):
            refine_distances(
                query.rows(query_rows.start, query_rows.stop),
                self.reference.rows(reference_rows.start, reference_rows.stop),
                distances,
                query_thresholds=distances.min(axis=1),
            )
            nearest[query_rows] = np.minimum(nearest[query_rows], distances.min(axis=1))
        return np.sqrt(nearest)
