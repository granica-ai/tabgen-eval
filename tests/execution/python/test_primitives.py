"""Unit tests for the statistical primitives (KS/TV complement, correlation, contingency)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.execution.python.primitives import (
    _reference_edges,
    contingency_similarity,
    correlation_similarity,
    equal_width_edges,
    ks_complement,
    quantile_edges,
    tv_complement,
)

pytestmark = pytest.mark.unit


def test_ks_complement_identical_is_one():
    s = pd.Series(np.arange(100.0))
    assert ks_complement(s, s) == pytest.approx(1.0)


def test_ks_complement_disjoint_is_low():
    a = pd.Series(np.zeros(100))
    b = pd.Series(np.ones(100))
    assert ks_complement(a, b) == pytest.approx(0.0, abs=1e-9)


def test_ks_complement_empty_is_nan():
    assert math.isnan(ks_complement(pd.Series([np.nan, np.nan]), pd.Series([1.0])))


def test_tv_complement_identical_is_one():
    s = pd.Series(["a", "b", "a", "c"])
    assert tv_complement(s, s) == pytest.approx(1.0)


def test_tv_complement_disjoint_is_zero():
    a = pd.Series(["a", "a", "a"])
    b = pd.Series(["b", "b", "b"])
    assert tv_complement(a, b) == pytest.approx(0.0)


def test_tv_complement_unseen_level_partial():
    a = pd.Series(["a"] * 8 + ["b"] * 2)
    b = pd.Series(["a"] * 6 + ["c"] * 4)
    score = tv_complement(a, b)
    assert 0.0 < score < 1.0


def test_correlation_similarity_matches_when_equal():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200))
    y = x * 2 + 1
    # Same correlation structure on both sides -> similarity 1.
    assert correlation_similarity(x, y, x, y) == pytest.approx(1.0)


def test_correlation_similarity_constant_is_nan():
    x = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([5.0, 5.0, 5.0])
    assert math.isnan(correlation_similarity(x, c, x, c))


def test_correlation_similarity_opposite_sign():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=300))
    a1, a2 = x, x  # corr ~ 1
    b1, b2 = x, -x  # corr ~ -1
    # |1 - (-1)| / 2 = 1 -> similarity 0.
    assert correlation_similarity(a1, a2, b1, b2) == pytest.approx(0.0, abs=1e-6)


def test_contingency_similarity_identical_is_one():
    a1 = pd.Series(["x", "y", "x", "y"] * 10)
    a2 = pd.Series(["p", "p", "q", "q"] * 10)
    assert contingency_similarity(
        a1, a2, a1, a2, col1_numeric=False, col2_numeric=False
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
@pytest.mark.parametrize("axis", [0, 1])
def test_contingency_excludes_actual_categorical_nulls(missing, axis):
    a = pd.Series(["a", "b", "a", "b"], dtype=object)
    b = pd.Series(["u", "u", "v", "v"], dtype=object)
    candidate_a = pd.Series(a.tolist() + [missing if axis == 0 else "a"], dtype=object)
    candidate_b = pd.Series(b.tolist() + [missing if axis == 1 else "u"], dtype=object)
    assert contingency_similarity(
        a, b, candidate_a, candidate_b, col1_numeric=False, col2_numeric=False
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("label", ["None", "<NA>", "NaT"])
def test_contingency_retains_literal_null_spellings(label):
    a = pd.Series(["a", "b", "a", "b"], dtype=object)
    b = pd.Series(["u", "u", "v", "v"], dtype=object)
    candidate_a = pd.Series(a.tolist() + [label], dtype=object)
    candidate_b = pd.Series(b.tolist() + ["u"], dtype=object)
    # The fifth row is a real new category, so its probability mass is counted.
    assert contingency_similarity(
        a, b, candidate_a, candidate_b, col1_numeric=False, col2_numeric=False
    ) == pytest.approx(0.8)


def test_contingency_similarity_numeric_binned():
    rng = np.random.default_rng(2)
    num = pd.Series(rng.normal(size=300))
    cat = pd.Series(rng.choice(["a", "b"], 300))
    score = contingency_similarity(num, cat, num, cat, col1_numeric=True, col2_numeric=False)
    assert score == pytest.approx(1.0)


def test_contingency_similarity_large_magnitude_no_crash():
    # A large-magnitude, narrow-range numeric column collapses linspace edges; must score
    # NaN (or a valid number) rather than raising "Bin edges must be unique" (T2).
    rng = np.random.default_rng(0)
    num = pd.Series(1e12 + rng.uniform(0, 1e-4, 200))
    cat = pd.Series(rng.choice(["a", "b"], 200))
    score = contingency_similarity(num, cat, num, cat, col1_numeric=True, col2_numeric=False)
    assert math.isnan(score) or 0.0 <= score <= 1.0


def test_quantile_edges_equal_frequency():
    # Skewed data: quantile bins each hold ~1/num_bins of the mass; equal-width bins do not.
    rng = np.random.default_rng(3)
    vals = rng.lognormal(sigma=1.5, size=10_000)
    edges = quantile_edges(vals, 10)
    assert edges is not None and edges.size == 11
    assert edges[0] == vals.min() and edges[-1] == vals.max()
    counts = pd.cut(vals, bins=edges, include_lowest=True).value_counts().to_numpy()
    assert counts.sum() == vals.size
    np.testing.assert_allclose(counts / vals.size, 0.1, atol=0.01)


def test_quantile_edges_ties_collapse_to_fewer_bins():
    # 90% of the mass on one value: coincident quantiles collapse, edges stay strictly increasing.
    vals = np.r_[np.zeros(900), np.arange(1.0, 101.0)]
    edges = quantile_edges(vals, 10)
    assert edges is not None
    assert 2 <= edges.size < 11
    assert np.all(np.diff(edges) > 0)
    pd.cut(vals, bins=edges, include_lowest=True)  # must not raise on duplicate edges


def test_quantile_edges_degenerate_inputs():
    assert quantile_edges(np.array([]), 10) is None
    assert quantile_edges(np.array([np.nan, np.inf, -np.inf]), 10) is None
    const = np.full(50, 7.0)
    np.testing.assert_array_equal(quantile_edges(const, 10), equal_width_edges(const, 10))
    # Non-finite values are ignored.
    edges = quantile_edges(np.array([1.0, 2.0, 3.0, np.nan, np.inf]), 2)
    np.testing.assert_array_equal(edges, [1.0, 2.0, 3.0])


def test_reference_edges_are_quantile():
    vals = np.random.default_rng(4).exponential(size=500)
    np.testing.assert_array_equal(_reference_edges(pd.Series(vals)), quantile_edges(vals, 10))


def test_contingency_similarity_quantile_resolves_skewed_column():
    # Heavy-tailed column: equal-width bins would put ~all rows in the first bin, so a candidate
    # that scrambles the bulk's association with the category would still score ~1. The quantile
    # bins the contingency path uses must resolve it.
    rng = np.random.default_rng(5)
    n = 4000
    num = pd.Series(rng.pareto(1.0, n))
    cat = pd.Series(np.where(num > num.median(), "hi", "lo"))
    shuffled = pd.Series(rng.permutation(cat.to_numpy()))
    kw = dict(col1_numeric=True, col2_numeric=False)
    assert contingency_similarity(num, cat, num, cat, **kw) == pytest.approx(1.0)
    assert contingency_similarity(num, cat, num, shuffled, **kw) < 0.6
