"""Small independent Shape oracles, CUDA protocol and bounded workspace checks."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import Shape
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def frames(dtype, *, cardinality=10):
    rng = np.random.default_rng(1381)
    tables = []
    for n in (263, 229):
        table = pd.DataFrame(rng.normal(size=(n, 3)).astype(dtype), columns=["x", "y", "z"])
        table["constant"] = np.full(n, 2.0, dtype=dtype)
        table["empty"] = np.nan
        table["category"] = rng.integers(cardinality, size=n).astype(str)
        table["literal"] = rng.choice(["nan", "a", "b"], n)
        table["missing"] = None
        table.loc[::13, "x"] = np.nan
        table.loc[::17, "y"] = np.nan
        table.loc[::19, "z"] = np.nan
        table.loc[::11, "category"] = None
        tables.append(table)
    return *tables, Schema(
        numeric=["x", "y", "z", "constant", "empty"],
        categorical=["category", "literal", "missing"],
    )


def oracle(a, b):
    """Evaluate source-selected values without invoking another backend."""
    scores = []
    for i in range(a.numeric.shape[1]):
        r, s = a.numeric[:, i].cpu().numpy(), b.numeric[:, i].cpu().numpy()
        r, s = r[np.isfinite(r)], s[np.isfinite(s)]
        scores.append(1 - ks_2samp(r, s).statistic if len(r) and len(s) else np.nan)
    for i in range(a.categorical.shape[1]):
        r, s = a.categorical[:, i].cpu().tolist(), b.categorical[:, i].cpu().tolist()
        r, s = [v for v in r if v >= 0], [v for v in s if v >= 0]
        cr, cs = Counter(r), Counter(s)
        scores.append(
            1 - 0.5 * sum(abs(cr[k] / len(r) - cs[k] / len(s)) for k in cr.keys() | cs.keys())
            if r and s
            else np.nan
        )
    finite = np.asarray(scores)[np.isfinite(scores)]
    return float(finite.mean()) if len(finite) else np.nan


@pytest.mark.parametrize("cardinality", [10, 1024, 15000])
def test_shape_matches_independent_oracle_and_cpu(torch, device, dtype, cardinality):
    r, s, schema = frames(dtype, cardinality=cardinality)
    metric = Shape(device=device, dtype=dtype)
    context = context_for(metric, r, s, schema)
    rng = np.random.default_rng(1382)
    rd, sd = rng.integers(0, len(r), (2, 103)), rng.integers(0, len(s), (2, 87))
    for ideal in (False, True):
        other = context.reference if ideal else context.candidate
        actual = context.compute(metric, rd, sd, ideal=ideal)["shape"]
        expected = [
            oracle(context.reference.gather(a), other.gather(b))
            for a, b in zip(rd, sd, strict=True)
        ]
        tolerance = 2e-6 if dtype == "float32" else 1e-12
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)
        cpu = context_for(Shape(device="cpu", dtype=dtype), r, s, schema)
        np.testing.assert_allclose(
            actual,
            cpu.compute(metric, rd, sd, ideal=ideal)["shape"],
            rtol=0,
            atol=tolerance,
            equal_nan=True,
        )


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed", "empty"])
def test_shape_degenerate_domains(torch, device, dtype, kind):
    schema = Schema(
        numeric=["x"] if kind != "categorical" else [],
        categorical=["c"] if kind != "numeric" else [],
    )
    r = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "c": [None, None, None]})
    if kind != "empty":
        r.loc[0, "x"], r.loc[0, "c"] = 4.0, "nan"
    s = r.iloc[[0, 0]].copy()
    metric = Shape(device=device, dtype=dtype)
    context = context_for(metric, r, s, schema)
    actual = context.compute(metric, np.array([[0, 1, 2]]), np.array([[0, 1]]))["shape"][0]
    assert np.isnan(actual) if kind == "empty" else actual == 1
    empty = context.compute(metric, np.empty((1, 0), dtype=np.int64), np.array([[0]]))["shape"][0]
    assert np.isnan(empty)


def test_shape_sparse_draw_from_actual_large_category_domain(torch, device, dtype):
    # Large global identities, tiny oracle draws: exercising a 15000-level
    # histogram does not require a large Python reference computation.
    r = pd.DataFrame({"c": np.arange(15_000).astype(str)})
    s = r.iloc[::-1].reset_index(drop=True)
    metric = Shape(device=device, dtype=dtype)
    context = context_for(metric, r, s, Schema(numeric=[], categorical=["c"]))
    assert context.inputs.cardinalities == (15_000,)
    rd, sd = np.array([[0, 1, 1, 14999]]), np.array([[0, 0, 1, 14999, 14999]])
    actual = context.compute(metric, rd, sd)["shape"][0]
    expected = oracle(context.reference.gather(rd[0]), context.candidate.gather(sd[0]))
    assert actual == pytest.approx(expected, abs=2e-6 if dtype == "float32" else 1e-12)


@pytest.mark.parametrize("bootstrap", [0, 2])
def test_shape_complete_reports_precision_and_stream(torch, device, dtype, bootstrap):
    r, s, schema = frames(dtype)
    before = (
        torch.cuda.current_device(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
    )
    stream = torch.cuda.Stream(device=device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16):
            actual = evaluate(
                r,
                s,
                schema,
                metrics=["shape"],
                options=make_options(
                    device=device,
                    dtype=dtype,
                    uncertainty="both" if bootstrap else "none",
                    n_replicates=max(1, bootstrap),
                ),
            )
        stream.synchronize()
        expected = evaluate(
            r,
            s,
            schema,
            metrics=["shape"],
            options=make_options(
                device="cpu",
                dtype=dtype,
                uncertainty="both" if bootstrap else "none",
                n_replicates=max(1, bootstrap),
            ),
        )
        a, b = actual["shape/shape"], expected["shape/shape"]
        assert a.flags == b.flags
        tolerance = 3e-6 if dtype == "float32" else 1e-12
        for field in (
            "value",
            "ideal",
            "ci_low",
            "ci_high",
            "ideal_ci_low",
            "ideal_ci_high",
            "generated_distribution",
            "reference_distribution",
        ):
            if getattr(a, field) is not None:
                np.testing.assert_allclose(
                    (
                        getattr(a, field).samples
                        if field.endswith("_distribution")
                        else getattr(a, field)
                    ),
                    (
                        getattr(b, field).samples
                        if field.endswith("_distribution")
                        else getattr(b, field)
                    ),
                    atol=tolerance,
                    rtol=0,
                    equal_nan=True,
                )
        assert actual.provenance["metric_execution"]["shape"]["implementation"] == "triton"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before[2]
    assert before == (
        torch.cuda.current_device(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
    )


def test_shape_workspace_bounds_and_chunk_invariance(torch, device):
    rng = np.random.default_rng(1383)
    schema = Schema(numeric=[f"n{i}" for i in range(12)], categorical=[f"c{i}" for i in range(12)])
    r = pd.DataFrame(rng.normal(size=(509, 12)), columns=schema.numeric)
    for c in schema.categorical:
        r[c] = rng.integers(15_000, size=len(r)).astype(str)
    s = r.sample(frac=1, random_state=0).reset_index(drop=True)
    rd = np.arange(len(r))[None, :]
    values = []
    for budget in (4, 64):
        metric = Shape(device=device, gpu_workspace_mb=budget)
        context = context_for(metric, r, s, schema)
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        values.append(context.compute(metric, rd, rd)["shape"][0])
        torch.cuda.synchronize(device)
        assert torch.cuda.max_memory_allocated(device) - baseline < budget * 1024**2
    assert values == [1, 1]
    metric = Shape(device=device, gpu_workspace_mb=1)
    context = context_for(metric, r, s, schema)
    with pytest.raises(MemoryError, match="Shape"):
        context.compute(metric, rd, rd)
