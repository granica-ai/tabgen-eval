"""Exactness guards for the vectorized Conditional/Fidelity primitives.

Compare searchsorted KS, integer-count TV, and conditional slice calculations
with independent SciPy KS and pandas frequency-table oracles. Assertions use
floating-point tolerances near machine precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from tabgen_eval import Schema
from tabgen_eval.execution.python.primitives import (
    NUM_BINS,
    contingency_similarity,
    contingency_similarity_codes,
    digitize_reference,
    factorize_shared,
    ks_complement,
    tv_complement,
)
from tabgen_eval.metrics.conditional import Conditional

pytestmark = pytest.mark.unit


def test_ks_complement_matches_scipy_statistic():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = rng.integers(0, 25, rng.integers(1, 400)).astype(float)  # ties on purpose
        y = rng.integers(0, 25, rng.integers(1, 400)).astype(float)
        ref = 1.0 - ks_2samp(x, y).statistic
        worst = max(worst, abs(ref - ks_complement(pd.Series(x), pd.Series(y))))
    assert worst < 1e-12, worst


def test_tv_complement_matches_pandas_total_variation():
    rng = np.random.default_rng(1)
    worst = 0.0
    cats = list("abcdefgh")
    for _ in range(200):
        a = pd.Series(rng.choice(cats, rng.integers(1, 300)))
        b = pd.Series(rng.choice(cats, rng.integers(1, 300)))
        pa = a.dropna().astype(str).value_counts(normalize=True)
        pb = b.dropna().astype(str).value_counts(normalize=True)
        idx = pa.index.union(pb.index)
        ref = 1.0 - 0.5 * float(
            np.abs(
                pa.reindex(idx, fill_value=0).to_numpy() - pb.reindex(idx, fill_value=0).to_numpy()
            ).sum()
        )
        worst = max(worst, abs(ref - tv_complement(a, b)))
    assert worst < 1e-12, worst


# --- a slow, obviously-correct reference for the whole conditional metric ----


def _slow_slice_shape(real: pd.DataFrame, syn: pd.DataFrame, cond: str, schema: Schema) -> float:
    numeric = set(schema.numeric)
    out = []
    for col in list(schema.numeric) + list(schema.categorical):
        if col == cond:
            continue
        if col in numeric:
            x = pd.to_numeric(real[col], errors="coerce").to_numpy(float)
            y = pd.to_numeric(syn[col], errors="coerce").to_numpy(float)
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            out.append(np.nan if x.size == 0 or y.size == 0 else 1.0 - ks_2samp(x, y).statistic)
        else:
            pa = real[col].value_counts(normalize=True)
            pb = syn[col].value_counts(normalize=True)
            support = pa.index.union(pb.index)
            out.append(
                1
                - 0.5
                * np.abs(
                    pa.reindex(support, fill_value=0) - pb.reindex(support, fill_value=0)
                ).sum()
            )
    arr = np.array(out, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _make_frame(n, seed):
    rng = np.random.default_rng(seed)
    p = np.array([0.55, 0.25, 0.12, 0.05, 0.03])  # skewed -> rare slices exist
    return pd.DataFrame(
        {
            "n0": rng.normal(0, 1, n),
            "n1": rng.gamma(2.0, 1.5, n),
            "n2": rng.integers(0, 8, n).astype(float),
            "c0": rng.choice([f"a{i}" for i in range(5)], size=n, p=p),
            "c1": rng.choice(list("xyz"), size=n),
        }
    )


def test_conditional_matches_slow_reference():
    schema = Schema(numeric=["n0", "n1", "n2"], categorical=["c0", "c1"])
    ref = _make_frame(2500, 0)
    syn = _make_frame(2000, 1)

    got = Conditional().compute(ref, syn, schema, rng=np.random.default_rng(0))

    # Recompute the four stats with the slow per-slice reference path.
    numeric = set(schema.numeric)
    conds = list(schema.numeric) + list(schema.categorical)
    cs_all, cs_rare, rc_all, rc_rare = [], [], [], []
    m = Conditional()
    for cond in conds:
        is_num = cond in numeric
        if is_num:
            low, high = ref[cond].min(), ref[cond].max()
            edges = np.unique(np.linspace(low, high, m.num_bins + 1))
            real_lab = pd.cut(ref[cond], bins=edges, labels=False, include_lowest=True)
            syn_lab = pd.cut(
                syn[cond].clip(edges[0], edges[-1]),
                bins=edges,
                labels=False,
                include_lowest=True,
            )
            values = sorted({v for v in real_lab.dropna().unique()})
        else:
            real_lab = ref[cond].astype(str)
            syn_lab = syn[cond].astype(str)
            values = sorted(real_lab.dropna().unique())
        total_real = int(real_lab.notna().sum())
        total_syn = int(syn_lab.notna().sum())
        rc, sc = real_lab.value_counts(), syn_lab.value_counts()
        sa, sr, ra, rr = [], [], [], []
        for v in values:
            n_real = int(rc.get(v, 0))
            if n_real < m.min_count:
                continue
            freq = n_real / total_real
            n_syn = int(sc.get(v, 0))
            rec = (n_syn / total_syn / freq) if (total_syn and freq) else np.nan
            if n_syn == 0:
                shape = 0.0
            else:
                shape = _slow_slice_shape(
                    ref[real_lab.to_numpy() == v], syn[syn_lab.to_numpy() == v], cond, schema
                )
            sa.append(shape)
            ra.append(rec)
            if freq < m.rare_threshold:
                sr.append(shape)
                rr.append(rec)
        cs_all.append(np.nanmean(sa) if sa else np.nan)
        cs_rare.append(np.nanmean(sr) if sr else np.nan)
        rc_all.append(np.nanmean(ra) if ra else np.nan)
        rc_rare.append(np.nanmean(rr) if rr else np.nan)

    assert got["cond_shape_all"] == pytest.approx(np.nanmean(cs_all), abs=1e-12)
    assert got["recovery_all"] == pytest.approx(np.nanmean(rc_all), abs=1e-12)
    assert got["cond_shape_rare"] == pytest.approx(np.nanmean(cs_rare), abs=1e-12)
    assert got["recovery_rare"] == pytest.approx(np.nanmean(rc_rare), abs=1e-12)


# --- parity: the two contingency implementations (pandas vs codes) -----------


def _axis_codes(a_ref: pd.Series, a_cand: pd.Series, is_numeric: bool):
    """One contingency axis as shared integer codes, exactly the way Conditional builds them:
    numeric -> real-anchored bins (``digitize_reference``), categorical -> ``factorize_shared``.
    Returns ``(ref_codes, cand_codes, n)`` or ``None`` for an unbinnable numeric range."""
    if is_numeric:
        return digitize_reference(
            pd.to_numeric(a_ref, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(a_cand, errors="coerce").to_numpy(dtype=float),
            NUM_BINS,
        )
    return factorize_shared(a_ref, a_cand)


def test_contingency_codes_matches_series():
    # ``contingency_similarity`` (pandas groupby/reindex-union) and ``contingency_similarity_codes``
    # (numpy bincount over the dense n_a*n_b grid) are *independent* implementations of the same TV,
    # so they need an explicit parity guard. Stress every type combo, with candidate numerics shifted
    # so some land OUTSIDE the reference range (the clip path), plus candidate-only categories and
    # NaNs on both sides -- exactly where the two paths could diverge.
    rng = np.random.default_rng(7)
    cats = list("abcdef")
    worst, checked = 0.0, 0

    def make(n: int, is_num: bool, shift: float) -> pd.Series:
        if is_num:
            s = pd.Series(rng.normal(shift, 1.5, n))
        else:
            # candidate (shift != 0) draws from a wider vocab -> some out-of-reference categories
            pool = cats if shift == 0.0 else cats + list("gh")
            s = pd.Series(rng.choice(pool, n)).astype(object)
        s[rng.random(n) < 0.05] = np.nan  # sprinkle NaNs on both sides
        return s

    for _ in range(400):
        t1, t2 = bool(rng.integers(2)), bool(rng.integers(2))
        n_ref, n_cand = int(rng.integers(30, 300)), int(rng.integers(30, 300))
        a1, a2 = make(n_ref, t1, 0.0), make(n_ref, t2, 0.0)
        b1, b2 = make(n_cand, t1, 0.8), make(n_cand, t2, -0.8)

        series = contingency_similarity(a1, a2, b1, b2, col1_numeric=t1, col2_numeric=t2)
        c1, c2 = _axis_codes(a1, b1, t1), _axis_codes(a2, b2, t2)

        if c1 is None or c2 is None:  # unbinnable numeric range -> the Series path scores NaN too
            assert np.isnan(series)
            continue
        ref_a, cand_a, n_a = c1
        ref_b, cand_b, n_b = c2
        codes = contingency_similarity_codes(ref_a, ref_b, cand_a, cand_b, n_a, n_b)

        if np.isnan(series) or np.isnan(codes):
            assert np.isnan(series) and np.isnan(codes)  # both degenerate, or neither
            continue
        worst = max(worst, abs(series - codes))
        checked += 1

    assert checked > 50, checked  # the loop actually compared finite scores
    assert worst < 1e-12, worst
