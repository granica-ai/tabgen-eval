"""Metamorphic relations: identity, noise-monotonicity, mode-drop, affine invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import PRDC, Conditional, Shape, Trend
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.metamorphic

SCHEMA = Schema(numeric=["a", "b"], categorical=["c"])


def _frame(n, shift=0.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(shift, noise, n),
            "b": rng.normal(shift, noise, n),
            "c": rng.choice(["x", "y", "z"], n),
        }
    )


def _rng():
    return np.random.default_rng(0)


def _ci_overlaps_ideal(s):
    """True iff the value CI overlaps the ideal CI; None if either is undefined."""
    xs = (s.ci_low, s.ci_high, s.ideal_ci_low, s.ideal_ci_high)
    if any(not np.isfinite(x) for x in xs):
        return None
    return s.ci_high >= s.ideal_ci_low and s.ideal_ci_high >= s.ci_low


@pytest.mark.calibration
def test_same_distribution_value_ci_overlaps_ideal(test_frame, syn_frame, schema):
    # generated ~ holdout (an independent draw from the same distribution) => the observed
    # comparison distribution equals the real-real ideal, so the value CI overlaps the ideal CI
    # (the cornerstone of the contract).
    #
    # Not ``test_frame.copy()``: the value's reference and candidate are drawn independently from
    # holdout and generated, so an exact copy would share ~half its rows across the two sides and
    # inflate the value above the ideal (whose halves are disjoint by construction).
    rep = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        options=make_options(uncertainty="both", n_replicates=80, seed=3),
    )
    for key in ["shape/shape", "trend/trend", "prdc/precision", "prdc/coverage"]:
        s = rep[key]
        assert _ci_overlaps_ideal(s) is True, key


def test_added_noise_worsens_shape():
    ref = _frame(600, seed=1)
    clean = _frame(600, noise=1.0, seed=2)
    noisy = _frame(600, noise=2.5, seed=3)
    s_clean = Shape().compute(ref, clean, SCHEMA, rng=_rng())["shape"]
    s_noisy = Shape().compute(ref, noisy, SCHEMA, rng=_rng())["shape"]
    assert s_clean > s_noisy


def test_mode_drop_hurts_coverage_more_than_precision():
    # Real has 3 categories + spread; syn collapses to one mode.
    ref = _frame(800, seed=1)
    syn = ref.copy()
    syn["c"] = "x"
    syn["a"] = np.random.default_rng(0).normal(0, 0.1, len(syn))
    out = PRDC().compute(ref, syn, SCHEMA, rng=_rng())
    # Collapsed syn fails to cover real (low coverage/recall).
    assert out["coverage"] < out["precision"]


def test_affine_scale_invariance_of_prdc():
    ref = _frame(400, seed=1)
    syn = _frame(400, seed=2)
    out = PRDC().compute(ref, syn, SCHEMA, rng=_rng())
    # Scale + shift the SAME numeric column in both frames -> min-max-scaled distances invariant.
    ref2, syn2 = ref.copy(), syn.copy()
    ref2["a"] = ref2["a"] * 100 + 5
    syn2["a"] = syn2["a"] * 100 + 5
    out2 = PRDC().compute(ref2, syn2, SCHEMA, rng=_rng())
    assert out["precision"] == pytest.approx(out2["precision"], abs=1e-6)
    assert out["coverage"] == pytest.approx(out2["coverage"], abs=1e-6)


def test_duplicate_syn_rows_reduce_coverage():
    # syn collapsed to a few unique rows (heavy duplication) covers fewer real modes.
    ref = _frame(800, seed=1)
    uniq = _frame(800, seed=2)
    dup = pd.concat([uniq.iloc[:80]] * 10, ignore_index=True)  # 800 rows, only 80 unique
    cov_uniq = PRDC().compute(ref, uniq, SCHEMA, rng=_rng())["coverage"]
    cov_dup = PRDC().compute(ref, dup, SCHEMA, rng=_rng())["coverage"]
    assert cov_dup < cov_uniq


@pytest.mark.calibration
def test_discrete_data_ideal_absorbs_ks_ties(schema):
    # Heavily-tied integer columns make KS conservative, but the real-real ideal is computed
    # the same way on the same discrete data, so it absorbs the bias: value CI overlaps ideal
    # CI when the generated set is an independent real draw.
    rng = np.random.default_rng(0)
    sch = Schema(numeric=["a", "b"])
    test = pd.DataFrame({"a": rng.integers(0, 8, 400), "b": rng.integers(0, 4, 400)})
    syn = pd.DataFrame({"a": rng.integers(0, 8, 400), "b": rng.integers(0, 4, 400)})
    rep = evaluate(
        test, syn, schema=sch, options=make_options(uncertainty="both", n_replicates=80, seed=1)
    )
    assert _ci_overlaps_ideal(rep["shape/shape"]) is True


def _simpson_frame(seed, *, break_within=False, n=3000):
    """Two groups with strong within-group a~b correlation; ``break_within`` shuffles b within
    each group, destroying the within-group association while preserving marginals and the
    between-group offsets — so the *pooled* correlation barely moves (Simpson's paradox)."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["A", "B"], n)
    a = np.where(g == "A", rng.normal(0, 1, n), rng.normal(6, 1, n))
    b = a + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"a": a, "b": b, "g": g})
    if break_within:
        for grp in ("A", "B"):
            mask = df["g"] == grp
            df.loc[mask, "b"] = rng.permutation(df.loc[mask, "b"].to_numpy())
    return df


def test_conditional_trend_catches_within_slice_break_global_trend_misses():
    # A within-subpopulation association break is large for Conditional Trend but nearly
    # invisible to global Trend (the between-group offsets keep the pooled correlation high).
    sch = Schema(numeric=["a", "b"], categorical=["g"])
    real = _simpson_frame(0)
    syn_good = _simpson_frame(1)
    syn_break = _simpson_frame(2, break_within=True)

    def _rng0():
        return np.random.default_rng(0)

    cond = Conditional()
    trend = Trend()
    d_cond = (
        cond.compute(real, syn_good, sch, rng=_rng0())["cond_trend_all"]
        - cond.compute(real, syn_break, sch, rng=_rng0())["cond_trend_all"]
    )
    d_global = (
        trend.compute(real, syn_good, sch, rng=_rng0())["trend"]
        - trend.compute(real, syn_break, sch, rng=_rng0())["trend"]
    )
    # Conditional Trend reacts to the break; global Trend barely does.
    assert d_cond > 0.3
    assert d_cond > 5 * d_global
