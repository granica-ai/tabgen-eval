"""Population controls for empirical score variation and targeted degradation.

These simulations check the behavior of raw paired score bands, without making
coverage claims for arbitrary metrics or treating overlap as a formal test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.python.distance import compute_prdc
from tabgen_eval.execution.python.encoding.features import build_feature_pair
from tabgen_eval.execution.runtime.sampling import fixed_draw, paired_draw_batches
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.calibration

SCHEMA = Schema(numeric=["a", "b"], categorical=["c"])


def _draw(n, rng, shift=0.0, corr=0.0):
    # `shift` moves column a's mean (degrades Shape, a marginal distance). `corr` injects a
    # correlation between a and b while leaving both marginals N(.,1) (degrades Trend, which is
    # location-invariant). The two corruptions are orthogonal: shift never touches Trend, corr
    # never touches the marginals Shape sees.
    a = rng.normal(shift, 1, n)
    z = rng.normal(0, 1, n)
    b = corr * (a - shift) + np.sqrt(max(1.0 - corr**2, 0.0)) * z if corr else z
    return pd.DataFrame({"a": a, "b": b, "c": rng.choice(["x", "y", "z"], n)})


def _ci_overlaps_ideal(s):
    """True iff the value CI overlaps the ideal CI; None if either interval is undefined."""
    xs = (s.ci_low, s.ci_high, s.ideal_ci_low, s.ideal_ci_high)
    if any(not np.isfinite(x) for x in xs):
        return None
    return s.ci_high >= s.ideal_ci_low and s.ideal_ci_high >= s.ci_low


def _overlap_rate(metric, key, *, shift=0.0, corr=0.0, trials=40, n=400, b=100):
    """Fraction of trials where the value CI overlaps the ideal CI.

    ``n=400`` -> the split ideal is evaluated at ``n_eff = n/2 = 200``, large enough to
    resolve a clearly-degraded generator from a real one.
    """
    rng = np.random.default_rng(11)
    hits = total = 0
    for t in range(trials):
        test = _draw(n, rng)
        syn = _draw(n, rng, shift=shift, corr=corr)
        s = evaluate(
            test,
            syn,
            schema=SCHEMA,
            metrics=[metric],
            options=make_options(uncertainty="both", n_replicates=b, seed=t),
        )[key]
        within = _ci_overlaps_ideal(s)
        if within is not None:
            total += 1
            hits += int(within)
    return hits / max(total, 1)


def test_calib_shape_overlap_and_power():
    # Shape is degraded by a marginal mean shift on column a.
    assert _overlap_rate("shape", "shape/shape", shift=0.0) >= 0.85  # ~1.0 observed
    assert _overlap_rate("shape", "shape/shape", shift=1.5) <= 0.25  # ~0.0 observed


def test_calib_trend_overlap_and_power():
    # Trend is degraded by an injected a-b correlation (a mean shift would be invisible to it).
    assert _overlap_rate("trend", "trend/trend", corr=0.0) >= 0.85  # ~1.0 observed
    assert _overlap_rate("trend", "trend/trend", corr=0.8) <= 0.25  # ~0.04 observed


def test_real_reference_draws_exclude_shared_source_positions():
    draw = fixed_draw(400, 400, seed=1)
    eligible = set()
    for batch in paired_draw_batches(
        draw, n_replicates=60, seed=1, batch_size=7, include_generated=True
    ):
        for ref, cand in zip(batch.reference, batch.real_candidate, strict=True):
            assert len(set(ref) | set(cand)) == 400
            assert not np.intersect1d(ref, cand).size
            eligible.update(ref)
    assert eligible == set(range(400))


def test_dup_inflation_magnitude_within_derived_bound():
    # Quantify the optimism (not just that the flag fires): the with-replacement (test, test)
    # ideal vs a disjoint-split ideal (two non-overlapping halves -> no shared rows -> no
    # distance-0 duplicates). The gap must stay within the analytic bound ~ 0.632 * (1 - v0)
    # derived for precision/recall/coverage.
    rng = np.random.default_rng(0)
    base = _draw(800, rng)
    feats, _ = build_feature_pair(base, base, SCHEMA)
    n = len(feats)
    for stat in ["precision", "recall", "coverage"]:
        with_repl, disjoint = [], []
        for r in range(20):
            g = np.random.default_rng(r)
            with_repl.append(
                compute_prdc(feats[g.integers(0, n, 400)], feats[g.integers(0, n, 400)], 5)[stat]
            )
            p = g.permutation(n)
            disjoint.append(compute_prdc(feats[p[:400]], feats[p[400:800]], 5)[stat])
        wr, dj = float(np.mean(with_repl)), float(np.mean(disjoint))
        allowed = 0.632 * (1.0 - dj) + 0.01  # derived bound + small slack
        assert abs(wr - dj) <= allowed, f"{stat}: |{wr:.4f} - {dj:.4f}| exceeds {allowed:.4f}"
