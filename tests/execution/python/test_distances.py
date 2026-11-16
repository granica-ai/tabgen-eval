"""Independent geometry and bounded-memory checks for shared distance kernels."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from tabgen_eval.execution.python.encoding import distances
from tabgen_eval.execution.python.encoding.distances import (
    NearestRecordIndex,
    distance_tiles,
    kth_neighbor_squared,
    prepare_features,
    refine_distances,
    squared_distances,
)
from tabgen_eval.execution.python.encoding.features import StructuredFeatures

pytestmark = pytest.mark.unit


def test_float64_direct_differences_preserve_nearby_large_coordinates():
    reference = np.array([[1e8, 1e8], [1e8 + 8, 1e8]], dtype=np.float32)
    actual = squared_distances(reference, reference)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [[0.0, 64.0], [64.0, 0.0]])
    np.testing.assert_array_equal(kth_neighbor_squared(reference, 1), [64.0, 64.0])
    # Promotion must preserve the feature encoding rather than refitting it.
    np.testing.assert_array_equal(prepare_features(reference).numeric, reference)


def test_compact_unknown_categories_match_independent_full_one_hot_geometry():
    numeric = np.array([[0.0], [0.25], [0.75], [1.0]], dtype=np.float32)
    codes = np.array([[0, 1], [1, -1], [-1, 0], [-1, -1]])
    compact = StructuredFeatures(numeric, codes, codes < 0)
    dense = np.column_stack(
        [numeric, *[(codes[:, column, None] == [0, 1]).astype(float) for column in range(2)]]
    )
    expected = cdist(dense.astype(float), dense.astype(float), "sqeuclidean")
    np.testing.assert_array_equal(squared_distances(compact, compact), expected)
    np.testing.assert_array_equal(kth_neighbor_squared(compact, 2), np.sort(expected)[:, 2])


@pytest.mark.parametrize("dimension", [10, 30])
def test_library_neighbor_search_retains_stable_float64_radii(dimension):
    # Accelerated search must preserve directly computed squared radii
    # even when coordinates have large offsets.
    rng = np.random.default_rng(9)
    features = (1e8 + rng.integers(0, 20, size=(40, dimension)) * 8).astype(np.float32)
    expected = np.partition(
        cdist(features.astype(float), features.astype(float), "sqeuclidean"), 5, axis=1
    )[:, 5]
    np.testing.assert_array_equal(kth_neighbor_squared(features, 5), expected)
    # Neighbors immediately around the rank threshold need exact ordering too.
    tied = np.array([[0.0], [1.0], [1.0 + 1e-12], [-1.0], [2.0]])
    expected = np.partition(cdist(tied, tied, "sqeuclidean"), 2, axis=1)[:, 2]
    np.testing.assert_array_equal(kth_neighbor_squared(tied, 2), expected)


@pytest.mark.parametrize("dimension", [2, 20])
def test_library_neighbor_queries_are_batched_with_one_rank_guard(monkeypatch, dimension):
    monkeypatch.setattr(distances, "_TILE_BYTES", 64 * 1024)
    monkeypatch.setattr(distances, "_MAX_QUERY_ROWS", 4)
    query_sizes = []
    fallbacks = []
    original_rank = distances._tiled_kth_neighbor_squared

    def observe_fallback(*args):
        fallbacks.append(len(args[0].numeric))
        return original_rank(*args)

    monkeypatch.setattr(distances, "_tiled_kth_neighbor_squared", observe_fallback)
    original = distances.NearestNeighbors.kneighbors

    def observe_query(model, query, *args, **kwargs):
        assert model.n_neighbors == (6 if dimension == 2 else 7)
        query_sizes.append(len(query))
        return original(model, query, *args, **kwargs)

    monkeypatch.setattr(distances.NearestNeighbors, "kneighbors", observe_query)
    features = np.random.default_rng(12).normal(size=(23, dimension)).astype(np.float32)
    expected = np.partition(
        cdist(features.astype(float), features.astype(float), "sqeuclidean"), 5, axis=1
    )[:, 5]
    np.testing.assert_array_equal(kth_neighbor_squared(features, 5), expected)
    assert sum(query_sizes) == len(features)
    assert max(query_sizes) <= 4
    assert fallbacks == []


def test_both_tile_axes_are_bounded_and_rankings_merge_across_reference_tiles(monkeypatch):
    monkeypatch.setattr(distances, "_TILE_BYTES", 128)
    monkeypatch.setattr(distances, "_MAX_QUERY_ROWS", 4)
    rng = np.random.default_rng(7)
    reference, query = rng.normal(size=(23, 20)), rng.normal(size=(17, 20))
    expected = cdist(query, reference, "sqeuclidean")
    reconstructed = np.empty_like(expected)
    for query_rows, reference_rows, tile in distance_tiles(query, reference):
        assert tile.nbytes <= 128
        reconstructed[query_rows, reference_rows] = tile
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-14, atol=2e-14)
    # k exceeds a single reference tile's width, requiring a true global merge.
    radii = np.partition(cdist(reference, reference, "sqeuclidean"), 5, axis=1)[:, 5]
    np.testing.assert_array_equal(kth_neighbor_squared(reference, 5), radii)
    np.testing.assert_array_equal(
        NearestRecordIndex(reference).query(query), np.sqrt(expected.min(axis=1))
    )


@pytest.mark.parametrize("dimension", [2, 20])
def test_nearest_queries_preserve_duplicate_weights_and_match_direct_distances(dimension):
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(30, dimension)).astype(np.float32)
    reference[1:4] = reference[0]
    query = np.vstack([reference[0], reference[0], reference[4]])
    index = NearestRecordIndex(reference)
    expected = cdist(query.astype(float), reference.astype(float)).min(axis=1)
    np.testing.assert_array_equal(index.query(query), expected)
    assert kth_neighbor_squared(reference, 2)[0] == 0.0


def test_invalid_and_empty_geometry_and_mismatched_layouts():
    empty = np.empty((0, 2))
    assert list(distance_tiles(empty, np.ones((3, 2)))) == []
    assert NearestRecordIndex(np.ones((3, 2))).query(empty).size == 0
    assert np.isnan(NearestRecordIndex(empty).query(np.ones((3, 2)))).all()
    assert np.isnan(kth_neighbor_squared(np.ones((2, 2)), 2)).all()
    assert np.isnan(kth_neighbor_squared(np.array([[np.nan], [0.0]]), 1)).all()
    assert np.isnan(NearestRecordIndex(np.array([[np.inf]])).query([[0.0]])).all()
    with pytest.raises(ValueError, match="layouts"):
        squared_distances(np.ones((3, 2)), np.ones((3, 4)))
    with pytest.raises(ValueError, match="positive"):
        kth_neighbor_squared(np.ones((3, 2)), 0)


def test_fast_tiles_refine_strict_boundaries_without_changing_the_comparison():
    rng = np.random.default_rng(5)
    query = rng.uniform(size=(30, 40)).astype(np.float32)
    reference = rng.uniform(size=(35, 40)).astype(np.float32)
    expected = cdist(query.astype(float), reference.astype(float), "sqeuclidean")
    actual = squared_distances(query, reference)
    # Each threshold passes through known equal-distance pairs. The kernel
    # must resolve equality directly rather than loosen the strict comparison.
    query_thresholds, reference_thresholds = expected[:, 0], expected[0]
    refine_distances(
        query,
        reference,
        actual,
        query_thresholds=query_thresholds,
        reference_thresholds=reference_thresholds,
    )
    np.testing.assert_array_equal(
        actual < query_thresholds[:, None], expected < query_thresholds[:, None]
    )
    np.testing.assert_array_equal(
        actual < reference_thresholds[None, :], expected < reference_thresholds[None, :]
    )
    np.testing.assert_array_equal(actual[:, 0], expected[:, 0])
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("dimension", [2, 40])
def test_cancellation_repairs_distances_and_uncertain_neighbor_rankings(dimension):
    # Float64 norm/dot expansion loses the small differences at this offset.
    features = 1e12 + np.random.default_rng(18).normal(size=(40, dimension))
    expected = cdist(features, features, "sqeuclidean")
    np.testing.assert_array_equal(squared_distances(features, features), expected)
    np.testing.assert_array_equal(
        kth_neighbor_squared(features, 5), np.partition(expected, 5, axis=1)[:, 5]
    )
    query = features[::3] + 0.25
    expected_nearest = np.sqrt(cdist(query, features, "sqeuclidean").min(axis=1))
    np.testing.assert_array_equal(NearestRecordIndex(features).query(query), expected_nearest)


def test_duplicate_rank_groups_use_direct_query_only_when_needed(monkeypatch):
    rng = np.random.default_rng(26)
    features = np.repeat(rng.normal(size=(20, 20)).astype(np.float32), 3, axis=0)
    queries = []
    original = distances.NearestNeighbors.kneighbors

    def observe(model, query, *args, **kwargs):
        queries.append(kwargs.get("n_neighbors", model.n_neighbors))
        return original(model, query, *args, **kwargs)

    def unexpected_fallback(*args):
        pytest.fail("ordinary duplicate groups should be resolved by the direct library query")

    monkeypatch.setattr(distances.NearestNeighbors, "kneighbors", observe)
    monkeypatch.setattr(distances, "_tiled_kth_neighbor_squared", unexpected_fallback)
    expected = np.partition(
        cdist(features.astype(float), features.astype(float), "sqeuclidean"), 4, axis=1
    )[:, 4]
    np.testing.assert_array_equal(kth_neighbor_squared(features, 4), expected)
    assert queries == [6, 5]
