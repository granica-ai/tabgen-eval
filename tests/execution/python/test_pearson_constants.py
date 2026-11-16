"""Exact decimal constants remain undefined Pearson pairs after finite filtering."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.python.conditional import compute_conditional
from tabgen_eval.execution.python.primitives import (
    correlation_similarity,
    correlation_similarity_arr,
)

pytestmark = pytest.mark.unit


def similarity(kind, arrays):
    if kind == "series":
        return correlation_similarity(*(pd.Series(values) for values in arrays))
    return correlation_similarity_arr(*arrays)


@pytest.mark.parametrize("kind", ["array", "series"])
@pytest.mark.parametrize("value,count", [(0.1, 15), (0.1, 20), (0.339244, 48)])
@pytest.mark.parametrize("constant_axis", range(4))
def test_exact_constants_are_undefined_on_either_axis_of_either_comparison(
    kind, value, count, constant_axis
):
    x = np.arange(count, dtype=float)
    arrays = [x, x**2, x, x**2]
    arrays[constant_axis] = np.full(count, value)
    assert np.isnan(similarity(kind, arrays))


@pytest.mark.parametrize("kind", ["array", "series"])
def test_constant_detection_follows_pairwise_nonfinite_filtering(kind):
    x = np.r_[np.full(15, 0.1), np.nan, np.inf, -np.inf]
    if kind == "series":
        # Series input accepts genuine missing cells; raw array kernels retain
        # their defensive nonfinite filtering below the input boundary.
        x[~np.isfinite(x)] = np.nan
    y = np.arange(len(x), dtype=float)
    assert np.isnan(similarity(kind, [x, y, y, y]))


@pytest.mark.parametrize("kind", ["array", "series"])
def test_distinct_near_constant_values_keep_a_defined_pearson_pair(kind):
    x = 0.1 + np.arange(20) * 1e-12
    y = np.arange(20, dtype=float)
    # Identical empirical pairs have similarity one regardless of their scale.
    assert similarity(kind, [x, y, x.copy(), y.copy()]) == pytest.approx(1.0)


@pytest.mark.parametrize("finite_pair", [False, True])
def test_conditional_trend_averages_only_mathematically_defined_pairs(finite_pair):
    n = 15
    reference = pd.DataFrame({"x": np.full(n, 0.1), "y": np.arange(n, dtype=float)})
    candidate = reference.copy()
    numeric = ["x", "y"]
    if finite_pair:
        # Among (x,y), (x,z), (y,z), only (y,z) is defined. Its correlations
        # are +1 and -1, so its similarity and the slice mean must both be zero.
        reference["z"] = reference["y"]
        candidate["z"] = -candidate["y"]
        numeric.append("z")
    reference["group"] = candidate["group"] = "a"
    result = compute_conditional(
        reference,
        candidate,
        Schema(numeric=numeric, categorical=["group"]),
        min_count=1,
        num_bins=2,
        rare_threshold=0.05,
        details=True,
    )
    group_index = result.details.metadata["column_names"].index("group")
    slices = result.details.tables["slices"].columns
    group_scores = slices["trend"][slices["column_index"] == group_index]
    column = result.details.tables["columns"].columns["cond_trend_all"][group_index]
    if finite_pair:
        np.testing.assert_allclose(group_scores, [0.0], atol=1e-15)
        assert column == pytest.approx(0.0, abs=1e-15)
    else:
        assert np.isnan(group_scores).all() and np.isnan(column)
