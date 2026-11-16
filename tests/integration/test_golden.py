"""Golden / snapshot tests: pin metric outputs on fixed frames (tolerance-based).

Compared with ``np.isclose`` tolerances, not exact equality, so last-ULP float differences
across platforms/BLAS don't make these flaky. Update the pinned values deliberately when a
metric's definition changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import PRDC, Conditional, Shape, Trend

pytestmark = pytest.mark.unit


def _frames():
    rng = np.random.default_rng(42)
    test = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 300),
            "b": rng.normal(2, 1, 300),
            "c": rng.choice(["x", "y", "z"], 300),
        }
    )
    syn = pd.DataFrame(
        {
            "a": rng.normal(0.3, 1, 300),
            "b": rng.normal(2, 1.1, 300),
            "c": rng.choice(["x", "y", "z"], 300),
        }
    )
    return test, syn, Schema(numeric=["a", "b"], categorical=["c"])


def _rng():
    return np.random.default_rng(0)


def test_golden_shape_trend():
    test, syn, sch = _frames()
    assert np.isclose(Shape().compute(test, syn, sch, rng=_rng())["shape"], 0.894444, atol=1e-4)
    assert np.isclose(Trend().compute(test, syn, sch, rng=_rng())["trend"], 0.867811, atol=1e-4)


def test_golden_prdc():
    test, syn, sch = _frames()
    out = PRDC().compute(test, syn, sch, rng=_rng())
    golden = {"precision": 0.99, "recall": 0.993333, "density": 0.988667, "coverage": 0.936667}
    for k, g in golden.items():
        assert np.isclose(out[k], g, atol=1e-4), f"{k}: {out[k]} != {g}"


def test_golden_conditional():
    test, syn, sch = _frames()
    out = Conditional().compute(test, syn, sch, rng=_rng())
    golden = {"cond_shape_all": 0.821384, "cond_trend_all": 0.659251, "recovery_all": 1.032151}
    for k, g in golden.items():
        assert np.isclose(out[k], g, atol=1e-4), f"{k}: {out[k]} != {g}"
    # These fixed 300-row frames have no rare slices, so the rare stats are NaN (NOT_APPLICABLE).
    for k in ("cond_shape_rare", "cond_trend_rare", "recovery_rare"):
        assert np.isnan(out[k]), f"{k}: {out[k]} expected NaN"
