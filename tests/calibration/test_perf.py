"""Performance / scale tests (slow -> marked ``perf``, run in the nightly lane).

These assert the package survives large inputs without OOM or runaway time: the chunked
pairwise-distance path scales, and generated pool size caps comparison resolution.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import PRDC
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.perf


@pytest.fixture(autouse=True)
def tensor_backend_dependency(request):
    if (
        getattr(request.node, "callspec", None)
        and request.node.callspec.params.get("backend") == "torch"
    ):
        pytest.importorskip("torch")


def test_prdc_chunked_distance_moderate_scale():
    # 5k x 5k cross-distances go through the chunked path; must finish quickly and stay finite.
    n = 5000
    rng = np.random.default_rng(0)
    test = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    syn = pd.DataFrame({"a": rng.normal(0.2, 1, n), "b": rng.normal(0, 1, n)})
    t0 = time.time()
    out = PRDC().compute(test, syn, Schema(numeric=["a", "b"]), rng=np.random.default_rng(0))
    assert time.time() - t0 < 30.0
    assert 0.0 <= out["precision"] <= 1.0 and 0.0 <= out["coverage"] <= 1.0


def test_generated_size_caps_large_holdout_resolution():
    # Both-side uncertainty uses half the smaller generated pool so membership can vary.
    # The full holdout remains eligible without increasing comparison resolution.
    n = 50_000
    rng = np.random.default_rng(0)
    test = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    syn = pd.DataFrame({"a": rng.normal(0.1, 1, n), "b": rng.normal(0, 1, n)})
    t0 = time.time()
    rep = evaluate(
        test,
        syn.iloc[:2000],
        schema=Schema(numeric=["a", "b"]),
        options=make_options(uncertainty="both", n_replicates=10),
    )
    assert time.time() - t0 < 60.0
    assert rep.provenance["n_eff"] == 1000
