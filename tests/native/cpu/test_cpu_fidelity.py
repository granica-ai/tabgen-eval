"""Compiled CPU Shape and Conditional numerical contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import Conditional, Shape

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("threads", [1, 4])
def test_shape_repeated_unequal_draws_missing_ties_and_sparse_global_vocabulary(
    torch, dtype, threads
):
    rng = np.random.default_rng(925)
    numeric = [f"n{i}" for i in range(6)]
    schema = Schema(numeric=numeric, categorical=["dense", "sparse"])
    real = pd.DataFrame(rng.integers(-3, 4, size=(6000, 6)).astype(dtype), columns=numeric)
    generated = pd.DataFrame(rng.integers(-3, 4, size=(6000, 6)).astype(dtype), columns=numeric)
    for table in [real, generated]:
        table["dense"] = rng.integers(10, size=6000).astype(str)
        table["sparse"] = np.arange(6000).astype(str)
        table.loc[::5, "n0"] = np.nan
        table.loc[::7, "n1"] = np.nan
        table["n2"] = np.nan
        table.loc[::11, "dense"] = None
        table.loc[::13, "sparse"] = None
    rd = rng.integers(0, 100, size=(3, 97), dtype=np.int64)
    sd = rng.integers(0, 120, size=(3, 113), dtype=np.int64)
    metric = Shape(device="cpu", dtype=dtype)
    context = context_for(metric, real, generated, schema, n_jobs=threads)
    actual = context.compute(metric, rd, sd)["shape"]
    expected = [
        Shape().compute(real.iloc[r], generated.iloc[s], schema, rng=None)["shape"]
        for r, s in zip(rd, sd, strict=True)
    ]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6 if dtype == "float32" else 1e-12)
    # Also exercise column parallelism for a single draw, independent of the
    # draw-parallel batch above and of the mostly unoccupied global vocabulary.
    single = context.compute(metric, rd[:1], sd[:1])["shape"]
    np.testing.assert_array_equal(single, actual[:1])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cardinality", [10, 1024, 15000])
def test_compiled_conditional_all_statistics_missing_rare_and_empty_slices(
    torch, dtype, cardinality
):
    rng = np.random.default_rng(822)
    r = pd.DataFrame(
        {
            "x": rng.normal(size=310),
            "y": rng.normal(size=310),
            "c": rng.integers(cardinality, size=310).astype(str),
        }
    )
    s = pd.DataFrame(
        {
            "x": rng.normal(size=270),
            "y": rng.normal(size=270),
            "c": rng.integers(cardinality, size=270).astype(str),
        }
    )
    for table in [r, s]:
        table[["x", "y"]] = table[["x", "y"]].astype(dtype)
        table.loc[::29, "x"] = np.nan
        table.loc[::31, "c"] = "nan"
        table.loc[::37, "c"] = np.nan
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    options = dict(min_count=1, rare_threshold=0.05)
    expected = Conditional(**options).compute(r, s, schema, rng=None)
    actual = Conditional(device="cpu", dtype=dtype, **options).compute(r, s, schema, rng=None)
    np.testing.assert_allclose(
        list(actual.values()),
        list(expected.values()),
        rtol=0,
        atol=3e-6 if dtype == "float32" else 1e-12,
        equal_nan=True,
    )
