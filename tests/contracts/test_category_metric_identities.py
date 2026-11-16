"""Independent synthetic fidelity and distance checks for typed category identities."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import PRDC, Conditional
from tests.support.distances import memorization

pytestmark = pytest.mark.unit


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def backend(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        pytest.importorskip("triton")
    return {"device": request.param}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_conditional_associations_remain_disjoint_across_integer_and_string_slices(backend, dtype):
    # Each binary conditioning column has two balanced slices. Within every
    # slice, the other pair occupies two reference cells and two disjoint
    # candidate cells, with unchanged marginals and slice frequencies.
    reference = pd.DataFrame(
        {
            "a": pd.Series([1, 1, "1", "1"] * 4, dtype=object),
            "b": ["u", "v", "u", "v"] * 4,
            "c": ["u", "v", "v", "u"] * 4,
        }
    )
    candidate = reference.copy(deep=True)
    candidate["c"] = ["v", "u", "u", "v"] * 4
    metric = Conditional(**backend, dtype=dtype, min_count=2, rare_threshold=0.75)
    scores = metric.compute(reference, candidate, Schema(categorical=["a", "b", "c"]), rng=None)
    assert scores == {
        "cond_shape_all": 1.0,
        "cond_shape_rare": 1.0,
        "cond_trend_all": 0.0,
        "cond_trend_rare": 0.0,
        "recovery_all": 1.0,
        "recovery_rare": 1.0,
    }


def distance_case():
    """Provide explicit one-hot coordinates without using package encoders."""
    reference = pd.DataFrame(
        {"x": [0, 2, 4, 8, 9], "c": pd.Series([1, 1.0, "1", True, None], dtype=object)}
    )
    candidate = pd.DataFrame(
        {
            "x": [0, 2, 4, 8, 12, 3, 5],
            "c": pd.Series([1.0, 1, "1", True, "unseen", np.int64(1), None], dtype=object),
        }
    )
    baseline = pd.DataFrame(
        {"x": [1, 3, 6, 10, 1], "c": pd.Series([1, 1.0, "unseen", True, None], dtype=object)}
    )
    identity = np.eye(4)

    def explicit(x, labels):
        categories = np.zeros((len(labels), 4))
        for row, label in enumerate(labels):
            if label >= 0:
                categories[row] = identity[label]
        return np.column_stack([np.asarray(x) / 8, categories])

    # Integer, float, string and boolean are four orthogonal coordinates.
    # Unknown observed labels have zero one-hot coordinates, and actual null
    # rows are excluded from distance geometry before fitting or scoring.
    arrays = [
        explicit([0, 2, 4, 8], [0, 1, 2, 3]),
        explicit([0, 2, 4, 8, 12, 3], [1, 0, 2, 3, -1, 0]),
        explicit([1, 3, 6, 10], [0, 1, -1, 3]),
    ]
    return reference, candidate, baseline, arrays, Schema(numeric=["x"], categorical=["c"])


def squared_distances(a, b):
    return ((a[:, None] - b[None, :]) ** 2).sum(axis=2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prdc_uses_four_typed_coordinates_and_reference_fitted_unknowns(backend, dtype):
    reference, candidate, _, (r, s, _), schema = distance_case()
    rr = np.partition(squared_distances(r, r), 1, axis=1)[:, 1]
    ss = np.partition(squared_distances(s, s), 1, axis=1)[:, 1]
    cross = squared_distances(r, s)
    inside = cross < rr[:, None]
    expected = {
        "precision": inside.any(axis=0).mean(),
        "recall": (cross < ss[None, :]).any(axis=1).mean(),
        "density": inside.sum() / len(s),
        "coverage": inside.any(axis=1).mean(),
    }
    actual = PRDC(**backend, dtype=dtype, nearest_k=1).compute(
        reference, candidate, schema, rng=None
    )
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, abs=2e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dcr_preserves_typed_coordinates_and_unknowns_in_both_query_pools(backend, dtype):
    reference, candidate, _, (r, s, _), schema = distance_case()
    generated_median = np.median(np.sqrt(squared_distances(s, r).min(axis=1)))
    holdout_median = generated_median
    device = backend["device"]
    outcome = memorization(reference, candidate, schema, device, dtype, holdout=candidate)
    assert outcome.values["median"] == pytest.approx(generated_median, abs=2e-7)
    assert outcome.diagnostics["holdout_median"] == pytest.approx(holdout_median, abs=2e-7)
    assert outcome.values["ratio"] == pytest.approx(generated_median / holdout_median, abs=2e-7)
    assert outcome.diagnostics["usable_rows"] == {"training": 4, "generated": 6, "holdout": 6}
