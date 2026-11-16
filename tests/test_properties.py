"""Property-based invariants (Hypothesis): range, permutation, determinism, symmetry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tabgen_eval import DiagnosticFlag, Schema
from tabgen_eval.metrics import Conditional, Shape, Trend
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.property

SCHEMA = Schema(numeric=["a", "b"], categorical=["c"])


@st.composite
def frames(draw, min_rows=30, max_rows=120):
    n = draw(st.integers(min_rows, max_rows))
    seed = draw(st.integers(0, 10_000))
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.choice(["x", "y", "z"], n),
        }
    )


def _rng():
    return np.random.default_rng(0)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(), cand=frames())
def test_shape_trend_in_unit_range(ref, cand):
    sh = Shape().compute(ref, cand, SCHEMA, rng=_rng())["shape"]
    tr = Trend().compute(ref, cand, SCHEMA, rng=_rng())["trend"]
    for v in (sh, tr):
        assert np.isnan(v) or 0.0 - 1e-9 <= v <= 1.0 + 1e-9


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(), cand=frames())
def test_conditional_trend_in_unit_range(ref, cand):
    out = Conditional().compute(ref, cand, SCHEMA, rng=_rng())
    for k in ("cond_trend_all", "cond_trend_rare"):
        v = out[k]
        assert np.isnan(v) or 0.0 - 1e-9 <= v <= 1.0 + 1e-9


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(min_rows=60), cand=frames(min_rows=60))
def test_conditional_trend_determinism(ref, cand):
    a = Conditional().compute(ref, cand, SCHEMA, rng=_rng())["cond_trend_all"]
    b = Conditional().compute(ref, cand, SCHEMA, rng=_rng())["cond_trend_all"]
    assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-12)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(), cand=frames())
def test_row_permutation_invariance(ref, cand):
    a = Shape().compute(ref, cand, SCHEMA, rng=_rng())["shape"]
    b = Shape().compute(
        ref.sample(frac=1.0, random_state=1).reset_index(drop=True),
        cand.sample(frac=1.0, random_state=2).reset_index(drop=True),
        SCHEMA,
        rng=_rng(),
    )["shape"]
    assert a == pytest.approx(b, abs=1e-9)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(), cand=frames())
def test_shape_symmetry(ref, cand):
    a = Shape().compute(ref, cand, SCHEMA, rng=_rng())["shape"]
    b = Shape().compute(cand, ref, SCHEMA, rng=_rng())["shape"]
    assert a == pytest.approx(b, abs=1e-9)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(min_rows=60), cand=frames(min_rows=60))
def test_full_pipeline_determinism(ref, cand):
    cfg = make_options(uncertainty="both", n_replicates=20, seed=9)
    a = evaluate(ref, cand, schema=SCHEMA, options=cfg).to_dataframe()
    b = evaluate(ref, cand, schema=SCHEMA, options=cfg).to_dataframe()
    assert a.fillna(-1).equals(b.fillna(-1))


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ref=frames(), cand=frames())
def test_column_permutation_invariance(ref, cand):
    # Reordering the columns of both frames must not change the result (align by name).
    a = Shape().compute(ref, cand, SCHEMA, rng=_rng())["shape"]
    cols = list(ref.columns)[::-1]
    b = Shape().compute(ref[cols], cand[cols], SCHEMA, rng=_rng())["shape"]
    assert a == pytest.approx(b, abs=1e-9)


def test_nondefault_index_does_not_change_results():
    # A non-default / non-unique DataFrame index must not affect results (index is reset).
    rng = np.random.default_rng(0)
    sch = Schema(numeric=["a"], categorical=["c"])
    ref = pd.DataFrame({"a": rng.normal(size=120), "c": rng.choice(["x", "y"], 120)})
    cand = pd.DataFrame({"a": rng.normal(0.2, size=120), "c": rng.choice(["x", "y"], 120)})
    a = Shape().compute(ref, cand, sch, rng=_rng())["shape"]
    ref2, cand2 = ref.copy(), cand.copy()
    ref2.index = [7] * len(ref2)  # non-unique index
    cand2.index = list(range(len(cand2), 0, -1))  # reversed index
    b = Shape().compute(ref2, cand2, sch, rng=_rng())["shape"]
    assert a == pytest.approx(b, abs=1e-9)


def test_saturation_produces_degenerate_ci_flag():
    # A stat pinned at a boundary (constant columns => Shape = 1.0 on every replicate) yields a
    # zero-width CI, flagged DEGENERATE_CI rather than crashing.
    df = pd.DataFrame({"a": [5.0] * 200, "b": [3.0] * 200})
    rep = evaluate(
        df,
        df.copy(),
        schema=Schema(numeric=["a", "b"]),
        options=make_options(uncertainty="both", n_replicates=50, seed=0),
    )
    assert DiagnosticFlag.DEGENERATE_CI in rep["shape/shape"].flags
