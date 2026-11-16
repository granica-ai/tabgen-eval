"""Per-metric unit tests: identical => near-optimum, degraded => worse."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, diagnose
from tabgen_eval.metrics import PRDC, Conditional, Shape, Trend

pytestmark = pytest.mark.unit

SCHEMA = Schema(numeric=["a", "b"], categorical=["c"])
RNG = np.random.default_rng(0)


def _frame(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(shift, 1, n),
            "b": rng.normal(shift, 1, n),
            "c": rng.choice(["x", "y", "z"], n),
        }
    )


def _rng():
    return np.random.default_rng(0)


def test_shape_identical_is_one():
    df = _frame(300)
    assert Shape().compute(df, df, SCHEMA, rng=_rng())["shape"] == pytest.approx(1.0)


def test_shape_degrades_with_shift():
    ref = _frame(500, 0.0, 1)
    near = _frame(500, 0.0, 2)
    far = _frame(500, 3.0, 3)
    s_near = Shape().compute(ref, near, SCHEMA, rng=_rng())["shape"]
    s_far = Shape().compute(ref, far, SCHEMA, rng=_rng())["shape"]
    assert s_near > s_far


def test_trend_identical_is_one():
    df = _frame(400)
    assert Trend().compute(df, df, SCHEMA, rng=_rng())["trend"] == pytest.approx(1.0, abs=1e-6)


def test_trend_not_applicable_single_column():
    schema1 = Schema(numeric=["a"])
    df = pd.DataFrame({"a": np.arange(50.0)})
    assert "trend" in Trend().not_applicable(df, df, schema1)


def test_prdc_identical_near_one():
    df = _frame(300)
    out = PRDC().compute(df, df, SCHEMA, rng=_rng())
    assert out["precision"] > 0.9
    assert out["coverage"] > 0.9


def test_prdc_insufficient_rows_nan():
    df = _frame(3)
    out = PRDC().compute(df, df, SCHEMA, rng=_rng())
    assert all(math.isnan(v) for v in out.values())


def test_prdc_uses_neighbor_constraint_without_a_universal_sample_floor():
    assert PRDC().n_min == 0
    assert PRDC(nearest_k=3).n_min == 0
    df = _frame(4)
    assert all(np.isnan(v) for v in PRDC(nearest_k=4).compute(df, df, SCHEMA, rng=_rng()).values())
    assert np.isfinite(PRDC(nearest_k=3).compute(df, df, SCHEMA, rng=_rng())["precision"])


@pytest.mark.parametrize("dim", [2, 10, 30])
@pytest.mark.parametrize("kind", ["gaussian", "onehot"])
def test_kth_nn_distance_matches_euclidean_reference(dim, kind):
    """Shared squared radii agree with independent direct float64 differences."""
    from scipy.spatial.distance import cdist

    from tabgen_eval.execution.python.encoding.distances import kth_neighbor_squared

    rng = np.random.default_rng(0)
    n, k = 500, 5
    if kind == "gaussian":
        features = rng.normal(0.0, 1.0, (n, dim)).astype(np.float32)
    else:
        features = np.zeros((n, dim), dtype=np.float32)
        for row in range(n):
            features[row, rng.integers(0, dim)] = 1.0
    direct = cdist(features.astype(float), features.astype(float), "sqeuclidean")
    expected = np.partition(direct, k, axis=1)[:, k]
    np.testing.assert_allclose(kth_neighbor_squared(features, k), expected, rtol=0, atol=1e-12)


def _cat_frame(n, cats_per_col, n_cols, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"c{j}": rng.choice(cats_per_col, n) for j in range(n_cols)},
    )


def test_structured_sqdist_matches_onehot_exact_with_oov():
    """The structured categorical distance equals the dense one-hot distance, incl. oov.

    Categorical-only, so the distances are integer-valued and the match is *exact* — this pins
    the ``2·[differ] − [one-side oov]`` formula (and the out-of-vocab correction the naive
    ``2·Hamming`` sketch gets wrong).
    """
    from scipy.spatial.distance import cdist

    from tabgen_eval.execution.python.encoding.distances import squared_distances
    from tabgen_eval.execution.python.encoding.features import fit_encoder

    schema = Schema(categorical=["c0", "c1"])
    ref = _cat_frame(200, list("abcde"), 2, seed=0)
    # candidate carries categories absent from the reference ("Z", "W") -> out-of-vocab
    cand = _cat_frame(150, list("abcdeZ"), 1, seed=1).assign(
        c1=np.random.default_rng(2).choice(list("xyzW"), 150)
    )
    enc = fit_encoder(ref, schema)
    oh_r, oh_c = enc.transform(ref).astype(np.float64), enc.transform(cand).astype(np.float64)
    onehot_sq = cdist(oh_r, oh_c, "sqeuclidean")
    structured_sq = squared_distances(enc.transform_structured(ref), enc.transform_structured(cand))
    assert np.allclose(onehot_sq, structured_sq, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("wide", [False, True])
def test_prdc_structured_matches_dense(wide):
    """End-to-end: the structured PRDC equals the dense one-hot PRDC at moderate cardinality."""
    from tabgen_eval.execution.python.distance import compute_prdc, compute_prdc_structured
    from tabgen_eval.execution.python.encoding.features import fit_encoder

    schema = Schema(numeric=["a", "b"], categorical=["c0", "c1"])

    def frame(n, seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "a": r.normal(0, 1, n),
                "b": r.normal(0, 1, n),
                "c0": r.choice([f"v{i}" for i in range(400 if wide else 30)], n),
                "c1": r.choice([f"w{i}" for i in range(400)] if wide else list("pqrs"), n),
            }
        )

    ref, cand = frame(600 if wide else 400, 0), frame(600 if wide else 400, 1)
    enc = fit_encoder(ref, schema)
    if wide:
        assert enc.onehot_dim > 512
    dense = compute_prdc(enc.transform(ref), enc.transform(cand), 5)
    structured = compute_prdc_structured(
        enc.transform_structured(ref), enc.transform_structured(cand), 5
    )
    for stat in dense:
        assert structured[stat] == pytest.approx(dense[stat], abs=1e-9)


def test_prdc_dense_and_compact_agree_at_duplicate_and_categorical_boundaries():
    from tabgen_eval.execution.python.distance import compute_prdc, compute_prdc_structured
    from tabgen_eval.execution.python.encoding.features import fit_encoder

    real = pd.DataFrame({"x": [0.0, 0.25, 0.5, 0.75] * 4, "c": list("abcd") * 4})
    fake = real.copy()
    fake.loc[:3, "c"] = "unknown"
    schema = Schema(numeric=["x"], categorical=["c"])
    encoder = fit_encoder(real, schema)
    dense = compute_prdc(encoder.transform(real), encoder.transform(fake), 5)
    compact = compute_prdc_structured(
        encoder.transform_structured(real), encoder.transform_structured(fake), 5
    )
    assert dense == compact


def test_prdc_reductions_merge_both_distance_tile_axes(monkeypatch):
    from scipy.spatial.distance import cdist

    from tabgen_eval.execution.python.distance import compute_prdc
    from tabgen_eval.execution.python.encoding import distances

    monkeypatch.setattr(distances, "_TILE_BYTES", 128)
    monkeypatch.setattr(distances, "_MAX_QUERY_ROWS", 4)
    rng = np.random.default_rng(11)
    real, fake, k = rng.normal(size=(23, 20)), rng.normal(size=(17, 20)), 5
    rr = np.partition(cdist(real, real, "sqeuclidean"), k, axis=1)[:, k]
    sr = np.partition(cdist(fake, fake, "sqeuclidean"), k, axis=1)[:, k]
    cross = cdist(real, fake, "sqeuclidean")
    inside = cross < rr[:, None]
    expected = {
        "precision": inside.any(axis=0).mean(),
        "recall": (cross < sr).any(axis=1).mean(),
        "density": inside.sum(axis=0).mean() / k,
        "coverage": (cross.min(axis=1) < rr).mean(),
    }
    assert compute_prdc(real, fake, k) == expected


def test_prdc_dispatches_to_structured_for_wide_onehot():
    """A high-cardinality frame routes through the structured path and gives a sane result."""
    from tabgen_eval.execution.python.distance import (
        _STRUCTURED_MIN_DIM,
        compute_prdc_structured,
    )
    from tabgen_eval.execution.python.encoding.features import fit_encoder

    schema = Schema(numeric=["a"], categorical=["c0", "c1"])

    def frame(n, seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "a": r.normal(0, 1, n),
                "c0": r.choice([f"u{i}" for i in range(400)], n),
                "c1": r.choice([f"w{i}" for i in range(400)], n),
            }
        )

    ref, cand = frame(600, 0), frame(600, 1)
    enc = fit_encoder(ref, schema)
    assert (
        enc.onehot_dim > _STRUCTURED_MIN_DIM
    )  # this frame really does trigger the structured path

    out = PRDC().compute(ref, cand, schema, rng=_rng())
    expected = compute_prdc_structured(
        enc.transform_structured(ref, dtype="float64"),
        enc.transform_structured(cand, dtype="float64"),
        PRDC().nearest_k,
    )
    assert out == expected  # PRDC.compute dispatched to exactly this path
    assert all(np.isfinite(v) for v in out.values())
    assert 0.0 <= out["precision"] <= 1.0 and 0.0 <= out["coverage"] <= 1.0


def test_conditional_identical_shape_high_recovery_one():
    # Larger frame so rare-category slices clear min_count.
    df = _frame(4000)
    out = Conditional().compute(df, df, SCHEMA, rng=_rng())
    assert out["cond_shape_all"] > 0.9
    assert out["recovery_all"] == pytest.approx(1.0, abs=0.05)


def test_conditional_empty_syn_slice_shape_zero():
    schema = Schema(numeric=["a"], categorical=["c"])
    ref = pd.DataFrame({"a": np.arange(100.0), "c": ["p"] * 50 + ["q"] * 50})
    # syn never produces category "q".
    syn = pd.DataFrame({"a": np.arange(100.0), "c": ["p"] * 100})
    out = Conditional(min_count=5).compute(ref, syn, schema, rng=_rng())
    # The real "q" slice the generator never populates contributes shape 0, so the
    # averaged conditional shape drops below a perfect 1.0.
    assert out["cond_shape_all"] < 1.0


def test_data_diagnostics_reports_raw_nan_rate():
    ref = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "c": ["x", "y", "x", "y"]})
    syn = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "c": ["x", "y", "x", "y"]})
    out = diagnose(ref, syn, schema=Schema(numeric=["a"], categorical=["c"]))
    assert out.inputs["train"]["missing_cell_rate"] > 0
    assert out.inputs["generated"]["missing_cell_rate"] == 0.0


# -- hard rule: compute() never raises on degenerate numeric columns (T1, T2) ---------------


def test_conditional_constant_numeric_conditioning_does_not_raise():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"k": [5.0] * 300, "c": rng.choice(list("xyz"), 300)})
    out = Conditional().compute(df, df.copy(), Schema(numeric=["k"], categorical=["c"]), rng=_rng())
    assert set(out) == {
        "cond_shape_all",
        "cond_shape_rare",
        "cond_trend_all",
        "cond_trend_rare",
        "recovery_all",
        "recovery_rare",
    }


def test_conditional_large_magnitude_numeric_does_not_raise():
    rng = np.random.default_rng(1)
    a = 1e12 + rng.uniform(0, 1e-3, 300)
    df = pd.DataFrame({"a": a, "c": rng.choice(list("xy"), 300)})
    # Must not raise "Bin edges must be unique".
    Conditional().compute(df, df.copy(), Schema(numeric=["a"], categorical=["c"]), rng=_rng())


def test_trend_large_magnitude_numeric_does_not_raise():
    rng = np.random.default_rng(2)
    a = 1e12 + rng.uniform(0, 1e-4, 300)
    df = pd.DataFrame({"a": a, "c": rng.choice(list("xy"), 300)})
    out = Trend().compute(df, df.copy(), Schema(numeric=["a"], categorical=["c"]), rng=_rng())
    assert math.isnan(out["trend"]) or 0.0 <= out["trend"] <= 1.0


# -- conditional Trend (within-slice pairwise association) -----------------------------------


def test_conditional_compute_returns_six_stats():
    df = _frame(4000)
    out = Conditional().compute(df, df, SCHEMA, rng=_rng())
    assert set(out) == {
        "cond_shape_all",
        "cond_shape_rare",
        "cond_trend_all",
        "cond_trend_rare",
        "recovery_all",
        "recovery_rare",
    }


def test_conditional_trend_identical_near_one():
    # Larger frame so numeric-tail slices clear min_count.
    df = _frame(4000)
    out = Conditional().compute(df, df, SCHEMA, rng=_rng())
    assert out["cond_trend_all"] > 0.9


def test_conditional_trend_empty_syn_slice_is_zero():
    # Three columns so within-slice trend is defined; syn never produces category "q", so that
    # retained real slice scores trend 0 and the average drops below a perfect 1.0.
    schema = Schema(numeric=["a", "b"], categorical=["c"])
    n = 200
    a = np.arange(float(n))
    ref = pd.DataFrame({"a": a, "b": a * 2.0, "c": ["p"] * (n // 2) + ["q"] * (n // 2)})
    syn = pd.DataFrame({"a": a, "b": a * 2.0, "c": ["p"] * n})
    out = Conditional(min_count=5).compute(ref, syn, schema, rng=_rng())
    assert out["cond_trend_all"] < 1.0


def test_conditional_trend_not_applicable_two_columns():
    # Trend needs a *pair* of other columns -> >= 3 total; shape/recovery only need >= 2.
    schema = Schema(numeric=["a"], categorical=["c"])
    df = pd.DataFrame({"a": np.arange(300.0), "c": list("xy") * 150})
    na = Conditional().not_applicable(df, df, schema)
    assert {"cond_trend_all", "cond_trend_rare"} <= na
    assert "cond_shape_all" not in na


def test_conditional_trend_constant_numeric_does_not_raise():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"k": [5.0] * 300, "a": rng.normal(0, 1, 300), "c": rng.choice(list("xyz"), 300)}
    )
    schema = Schema(numeric=["k", "a"], categorical=["c"])
    out = Conditional().compute(df, df.copy(), schema, rng=_rng())
    assert math.isnan(out["cond_trend_all"]) or 0.0 <= out["cond_trend_all"] <= 1.0
