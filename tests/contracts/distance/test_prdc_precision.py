"""Independent PRDC coordinate-precision and strict-neighborhood contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.python import distance
from tabgen_eval.metrics import PRDC
from tests.support.distances import distance_matrices

pytestmark = pytest.mark.unit


def _oracle(reference, candidate, schema, dtype, k):
    """Build explicit typed one-hot geometry and rank full direct-distance matrices."""
    matrix, _ = distance_matrices(
        reference, [candidate], schema, dtype, scaling_dtype="float64", accumulation_dtype="float64"
    )
    rr = np.partition(matrix(0, 0), k, axis=1)[:, k]
    ff = np.partition(matrix(1, 1), k, axis=1)[:, k]
    cross = matrix(0, 1)
    inside = cross < rr[:, None]
    return {
        "precision": float(inside.any(axis=0).mean()),
        "recall": float((cross < ff[None, :]).any(axis=1).mean()),
        "density": float(inside.sum(axis=0).mean() / k),
        "coverage": float(inside.any(axis=1).mean()),
    }


@pytest.mark.parametrize("compact", [False, True])
def test_default_and_requested_coordinate_precision_preserve_strict_near_boundaries(
    compact, monkeypatch
):
    monkeypatch.setattr(distance, "_STRUCTURED_MIN_DIM", 0 if compact else 512)
    reference = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    candidate = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0]) - 2**-30})
    schema = Schema(numeric=["x"])
    expected64 = dict(precision=1.0, recall=1.0, density=5 / 3, coverage=1.0)
    expected32 = dict(precision=2 / 3, recall=2 / 3, density=2 / 3, coverage=2 / 3)
    assert _oracle(reference, candidate, schema, "float64", 1) == expected64
    assert _oracle(reference, candidate, schema, "float32", 1) == expected32
    assert distance.prdc(reference, candidate, schema, 1) == expected64
    for dtype, expected in [("float32", expected32), ("float64", expected64)]:
        actual = PRDC(dtype=dtype, nearest_k=1).compute(reference, candidate, schema, rng=None)
        assert actual == expected
    # A genuine equal-distance boundary remains excluded at both precisions.
    boundary = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    for dtype in ("float32", "float64"):
        assert distance.prdc(reference, boundary, schema, 1, dtype=dtype) == expected32


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_precision_retains_typed_categories_duplicates_and_complete_row_filtering(
    compact, dtype, monkeypatch
):
    monkeypatch.setattr(distance, "_STRUCTURED_MIN_DIM", 0 if compact else 512)
    reference = pd.DataFrame(
        {
            "x": [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, np.nan, 100.0],
            "c": pd.Series([1, 1, 1.0, "1", "1", "a", "a", None], dtype=object),
        }
    )
    candidate = pd.DataFrame(
        {
            "x": [0.0, 0.0, 0.5, 1.0, 2.0, 1.0 + 2**-30, np.nan, 100.0],
            "c": pd.Series([1, 1, 1.0, "1", "unknown", "a", 1, None], dtype=object),
        }
    )
    schema = Schema(numeric=["x"], categorical=["c"])
    expected = _oracle(reference, candidate, schema, dtype, 2)
    actual = PRDC(dtype=dtype, nearest_k=2).compute(reference, candidate, schema, rng=None)
    assert actual == expected
    # Exact duplicates give a zero-radius neighborhood for k=1, so strict < excludes them.
    duplicates = pd.DataFrame({"x": [0.0, 0.0, 1.0, 1.0]})
    assert distance.prdc(duplicates, duplicates, Schema(numeric=["x"]), 1, dtype=dtype) == {
        "precision": 0.0,
        "recall": 0.0,
        "density": 0.0,
        "coverage": 0.0,
    }


@pytest.mark.parametrize("compact", [False, True])
def test_python_and_native_float64_agree_at_exact_and_near_boundaries(compact, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(distance, "_STRUCTURED_MIN_DIM", 0 if compact else 512)
    reference = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    schema = Schema(numeric=["x"])
    for offset in (0.0, -(2**-30), 2**-30):
        candidate = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0]) + offset})
        expected = _oracle(reference, candidate, schema, "float64", 1)
        reference_result = distance.prdc(reference, candidate, schema, 1, dtype="float64")
        actual = PRDC(device="cpu", dtype="float64", nearest_k=1).compute(
            reference, candidate, schema, rng=None
        )
        assert reference_result == expected
        assert actual == expected
