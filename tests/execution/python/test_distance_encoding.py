"""Reference-fitted precision and invalid-value contracts for Python distances."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from tabgen_eval import Schema
from tabgen_eval.execution.python import distance
from tabgen_eval.execution.python.encoding.features import fit_encoder
from tests.support.distances import memorization

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dimension", [1, 16])
def test_dcr_preserves_float64_near_copies_and_exact_copies(dimension):
    columns = [f"x{i}" for i in range(dimension)]
    reference = pd.DataFrame([np.zeros(dimension), np.ones(dimension)], columns=columns)
    candidate = pd.DataFrame([np.ones(dimension)], columns=columns)
    candidate.iloc[0, 0] += 1e-9
    schema = Schema(numeric=columns)
    actual = memorization(reference, candidate, schema)
    assert actual.values["median"] == candidate.iloc[0, 0] - 1.0 > 0
    assert actual.diagnostics["holdout_median"] is None
    exact = memorization(reference, reference.iloc[[1, 1, 0]], schema)
    assert exact.values["median"] == 0.0


@pytest.mark.parametrize("invalid_pool", ["training", "generated"])
def test_invalid_numeric_pool_raises_instead_of_creating_observed_zero(invalid_pool):
    schema = Schema(numeric=["x"])
    pools = {
        "training": pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        "generated": pd.DataFrame({"x": [0.5, 1.5]}),
    }
    pools[invalid_pool] = pd.concat(
        [pools[invalid_pool], pd.DataFrame({"x": ["bad"]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="declared numeric"):
        memorization(*pools.values(), schema)


@pytest.mark.parametrize("pool", ["reference", "candidate"])
@pytest.mark.parametrize("compact", [False, True])
def test_prdc_rejects_invalid_numeric_values(pool, compact, monkeypatch):
    monkeypatch.setattr(distance, "_STRUCTURED_MIN_DIM", 0 if compact else 512)
    schema = Schema(numeric=["x"], categorical=["c"])
    frames = {
        "reference": pd.DataFrame({"x": [0.0, 1.0, 2.0], "c": ["a", "b", "a"]}),
        "candidate": pd.DataFrame({"x": [0.5, 1.5, 2.5], "c": ["a", "b", "a"]}),
    }
    frames[pool]["x"] = frames[pool]["x"].astype(object)
    frames[pool].loc[0, "x"] = "bad"
    with pytest.raises(ValueError, match="declared numeric"):
        distance.prdc(*frames.values(), schema, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_and_compact_encodings_preserve_typed_categories_and_precision(dtype):
    reference = pd.DataFrame({"x": [0.0, 1.0, 2.0], "c": pd.Series([1, 1.0, "1"], dtype=object)})
    candidate = pd.DataFrame(
        {"x": [1 + 1e-9] * 4, "c": pd.Series([1, 1.0, "1", "unknown"], dtype=object)}
    )
    encoder = fit_encoder(reference, Schema(numeric=["x"], categorical=["c"]))
    dense = encoder.transform(candidate, dtype=dtype)
    compact = encoder.transform_structured(candidate, dtype=dtype)
    assert dense.dtype == compact.numeric.dtype == dtype
    np.testing.assert_array_equal(dense[:, :1], compact.numeric)
    np.testing.assert_array_equal(dense[:, 1:], np.vstack([np.eye(3), np.zeros(3)]))
    np.testing.assert_array_equal(compact.codes[:, 0], [0, 1, 2, -1])
    np.testing.assert_array_equal(compact.oov[:, 0], [False, False, False, True])
    assert encoder.transform(candidate).dtype == np.float32
    assert encoder.transform_structured(candidate).numeric.dtype == np.float32


@pytest.mark.parametrize("compact", [False, True])
def test_valid_prdc_matches_independent_float32_geometry(compact, monkeypatch):
    monkeypatch.setattr(distance, "_STRUCTURED_MIN_DIM", 0 if compact else 512)
    rng = np.random.default_rng(2026)
    real_values, fake_values = rng.uniform(size=(11, 2)), rng.uniform(size=(13, 2))
    real_codes, fake_codes = rng.integers(0, 3, 11), rng.integers(-1, 3, 13)
    reference = pd.DataFrame(real_values, columns=["x", "y"]).assign(c=real_codes)
    candidate = pd.DataFrame(fake_values, columns=["x", "y"]).assign(c=fake_codes)
    mins, ranges = real_values.min(axis=0), np.ptp(real_values, axis=0)
    real = np.column_stack([(real_values - mins) / ranges, np.eye(3)[real_codes]]).astype(
        np.float32
    )
    fake = np.column_stack(
        [(fake_values - mins) / ranges, fake_codes[:, None] == np.arange(3)]
    ).astype(np.float32)
    cross = cdist(real.astype(float), fake.astype(float), "sqeuclidean")
    real_radius = np.sort(cdist(real.astype(float), real.astype(float), "sqeuclidean"))[:, 2]
    fake_radius = np.sort(cdist(fake.astype(float), fake.astype(float), "sqeuclidean"))[:, 2]
    real_balls = cross < real_radius[:, None]
    expected = {
        "precision": real_balls.any(axis=0).mean(),
        "recall": (cross < fake_radius[None, :]).any(axis=1).mean(),
        "density": real_balls.sum(axis=0).mean() / 2,
        "coverage": (cross.min(axis=1) < real_radius).mean(),
    }
    actual = distance.prdc(
        reference,
        candidate,
        Schema(numeric=["x", "y"], categorical=["c"]),
        2,
        dtype="float32",
    )
    assert actual == expected


@pytest.mark.parametrize("compact", [False, True])
def test_excluded_columns_do_not_remove_records_from_python_distances(compact, monkeypatch):
    monkeypatch.setattr(distance, "_STRUCTURED_MIN_DIM", 0 if compact else 512)
    reference = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    candidate = pd.DataFrame({"x": [0.25, 1.25, 2.25]})
    schema = Schema(numeric=["x"])
    expected = distance.prdc(reference, candidate, schema, 1)
    np.testing.assert_equal(
        distance.prdc(reference.assign(ignored=None), candidate.assign(ignored=None), schema, 1),
        expected,
    )
    outcome = memorization(reference.assign(ignored=None), candidate.assign(ignored=None), schema)
    assert outcome.diagnostics["usable_rows"] == {"training": 3, "generated": 3, "holdout": None}
