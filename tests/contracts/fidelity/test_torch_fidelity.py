"""Independent fidelity oracles and complete tensor Trend report contracts."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import Trend
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def frames(kind="mixed", n=263, cardinality=10):
    rng = np.random.default_rng(721)
    numeric = [f"n{i}" for i in range(3)] if kind != "categorical" else []
    categorical = [f"c{i}" for i in range(3)] if kind != "numeric" else []
    tables = []
    for rows in [n, n - 17]:
        frame = pd.DataFrame({c: rng.normal(size=rows) for c in numeric})
        for c in categorical:
            frame[c] = rng.integers(cardinality, size=rows).astype(str)
            frame.loc[::29, c] = "nan"
            frame.loc[::37, c] = np.nan
        for i, c in enumerate(numeric):
            frame.loc[i::31, c] = np.nan
        tables.append(frame)
    if numeric:
        tables[1].loc[::19, numeric[0]] = np.nan
    return *tables, Schema(numeric=numeric, categorical=categorical)


@pytest.mark.parametrize(
    "kind,cardinality",
    [("numeric", 10)] + [(kind, n) for kind in ["categorical", "mixed"] for n in [10, 15000]],
)
def test_trend_matches_dataframe_reference(torch, device, dtype, kind, cardinality):
    r, s, schema = frames(kind, cardinality=cardinality)
    for table in [r, s]:
        table[schema.numeric] = table[schema.numeric].astype(dtype)
    expected = Trend().compute(r, s, schema, rng=None)["trend"]
    actual = Trend(device=device, dtype=dtype).compute(r, s, schema, rng=None)["trend"]
    assert actual == pytest.approx(expected, abs=3e-6 if dtype == "float32" else 1e-12)


def independent_correlations(values):
    p = values.shape[1]
    out = np.full((p, p), np.nan)
    for i, j in combinations(range(p), 2):
        pair = values[np.isfinite(values[:, i]) & np.isfinite(values[:, j])][:, [i, j]]
        if len(pair) < 2:
            continue
        # Extended precision is an independent oracle, not an execution mode.
        pair = pair.astype(np.longdouble)
        pair -= pair[0].copy()
        pair -= pair.mean(axis=0)
        norms = (pair * pair).sum(axis=0)
        if (norms > 0).all():
            out[i, j] = out[j, i] = float((pair[:, 0] * pair[:, 1]).sum() / np.sqrt(norms.prod()))
    return out


def test_correlations_pairwise_missing_constants_and_high_offsets(torch, device, dtype):
    from tabgen_eval.execution.torch.fidelity.primitives import correlations

    rng = np.random.default_rng(912)
    x = rng.normal(size=(211, 7)).astype(dtype)
    x[:, 1] = x[:, 0] * 2
    x[:, 2] = 5
    x[:, 3] += 1e12 if dtype == "float64" else 1e4
    x[3:, 4] = np.nan
    x[:3, 0] = 1  # This pair is constant only on the overlapping rows.
    x[:, 5] = np.nan
    x[::3, 6] = np.inf
    expected = independent_correlations(x)
    actual = correlations(torch.tensor(x, device=device)).cpu().numpy()
    upper = np.triu_indices(x.shape[1], k=1)
    np.testing.assert_allclose(
        actual[upper],
        expected[upper],
        rtol=0,
        atol=3e-6 if dtype == "float32" else 1e-12,
        equal_nan=True,
    )
    assert np.isnan(actual[0, 4])


def test_quantile_edges_ties_and_right_closed_reference_bins(torch, device, dtype):
    from tabgen_eval.execution.python.primitives import _bin_codes
    from tabgen_eval.execution.python.primitives import quantile_edges as reference_edges
    from tabgen_eval.execution.torch.fidelity.primitives import bin_codes, quantile_edges

    kind = getattr(torch, dtype)
    for values in [[0, 0, 0, 1, 1, 3, 8, 8], [7] * 12, [1e12] * 12, [np.nan, np.inf]]:
        source = np.asarray(values, dtype=dtype)
        edges = quantile_edges(torch.tensor(source, device=device))
        if not np.isfinite(source).any():
            assert edges is None
            continue
        expected = reference_edges(source.astype(np.float64), 10)
        np.testing.assert_allclose(
            edges.cpu().numpy(), expected, rtol=3e-6 if dtype == "float32" else 1e-15
        )
        # Compare membership to an independent search using the actual selected-
        # precision edges, including ties, minimum and candidate clipping.
        query = torch.cat(
            (edges, torch.tensor([-np.inf, np.inf, np.nan], device=device, dtype=kind))
        )
        got = bin_codes(query, edges, clip=True, clip_infinite=True).cpu().numpy()
        q = query.cpu().numpy().astype(np.float64)
        limits = edges.cpu().numpy().astype(np.float64)
        upper = float(edges[-1].item())
        finite_q = np.clip(q, limits[0], upper)
        wanted = np.searchsorted(limits, finite_q, side="left") - 1
        wanted[finite_q == limits[0]] = 0
        wanted[np.isnan(q)] = -1
        np.testing.assert_array_equal(got, wanted)
        dropped = bin_codes(query, edges, clip=True).cpu().numpy()
        np.testing.assert_array_equal(dropped[-3:], [-1, -1, -1])
        regular = query[:-3].cpu().numpy()
        np.testing.assert_array_equal(
            bin_codes(query[:-3], edges).cpu().numpy(), _bin_codes(regular, limits, clip=False)
        )


def test_complete_correlations_use_stable_selected_precision(torch, device, dtype):
    from tabgen_eval.execution.torch.fidelity.primitives import correlations

    rng = np.random.default_rng(147)
    values = rng.normal(size=(173, 5)).astype(dtype)
    values[:, 1] = 2 * values[:, 0] + 1
    values[:, 2] += 1e12 if dtype == "float64" else 1e4
    values[:, 3] = 1
    # Opposite finite extremes exercise scaling before an overflowing shift.
    extreme = np.finfo(np.dtype(dtype)).max * 0.75
    values[:, 4] = np.where(values[:, 0] > 0, extreme, -extreme)
    expected = independent_correlations(values)
    actual = correlations(torch.tensor(values, device=device)).cpu().numpy()
    upper = np.triu_indices(5, k=1)
    np.testing.assert_allclose(
        actual[upper],
        expected[upper],
        atol=3e-6 if dtype == "float32" else 1e-12,
        rtol=0,
        equal_nan=True,
    )


def counter_similarity(a, b, pair):
    def counts(table):
        return Counter(tuple(row) for row in table[:, pair] if (row >= 0).all())

    ca, cb = counts(a), counts(b)
    na, nb = sum(ca.values()), sum(cb.values())
    if not na or not nb:
        return np.nan
    return 1 - 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in ca.keys() | cb.keys())


@pytest.mark.parametrize("levels", [[3, 7, 10], [10, 1024, 15000]])
def test_contingency_batches_match_independent_occupied_cell_counts(torch, device, dtype, levels):
    from tabgen_eval.execution.torch.fidelity.primitives import contingency_scores

    rng = np.random.default_rng(211)
    a = np.column_stack([rng.integers(c, size=389) for c in levels])
    b = np.column_stack([rng.integers(c, size=327) for c in levels])
    a[::17, 0] = -1
    b[::23, 1] = -1
    expected = [counter_similarity(a, b, pair) for pair in combinations(range(3), 2)]
    prototype = torch.empty((), device=device, dtype=getattr(torch, dtype))
    actual = contingency_scores(
        torch.tensor(a, device=device),
        torch.tensor(b, device=device),
        levels,
        0,
        prototype,
        1 << 20,
    )
    # Group scheduling can reorder pairs. Compare the per-pair scores as a multiset.
    np.testing.assert_allclose(
        np.sort(actual.cpu().numpy()),
        np.sort(expected),
        rtol=0,
        atol=2e-6 if dtype == "float32" else 1e-12,
    )


def test_shared_ks_tv_primitives_match_independent_distributions(torch, device, dtype):
    from tabgen_eval.execution.torch.fidelity.primitives import ks_complement, tv_complement_codes

    kind = getattr(torch, dtype)
    a = torch.tensor([0, 0, 1, 3, float("nan"), float("inf")], device=device, dtype=kind)
    b = torch.tensor([0, 2, 2, 2, 3], device=device, dtype=kind)
    expected = 1 - ks_2samp([0, 0, 1, 3], [0, 2, 2, 2, 3]).statistic
    assert ks_complement(a, b).item() == pytest.approx(expected, abs=1e-6)
    ca = torch.tensor([0, 0, 1, -1], device=device)
    cb = torch.tensor([0, 2, 2, -1], device=device)
    assert tv_complement_codes(ca, cb, 15000, a.new_empty(())).item() == pytest.approx(
        1 / 3, abs=1e-6
    )
    assert torch.isnan(ks_complement(a[:0], b))
    assert torch.isnan(tv_complement_codes(ca[:0], cb, 15000, a.new_empty(())))


@pytest.mark.parametrize(
    "op",
    ["fidelity_gram", "finite_pair_counts", "joint_keys", "joint_dense_counts", "joint_sorted_tv"],
)
def test_fidelity_operator_schema_and_fake_contracts(torch, device, dtype, op):
    from tabgen_eval.execution.torch.fidelity import ops as fidelity_ops

    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]], device=device, dtype=getattr(torch, dtype)
    )
    c = torch.tensor([[0, 1], [1, 0], [0, 1]], device=device)
    pairs, widths = torch.tensor([[0, 1]], device=device), torch.tensor([2], device=device)
    if op == "fidelity_gram":
        args = (x, x)
    elif op == "finite_pair_counts":
        args = (torch.isfinite(x),)
    elif op == "joint_keys":
        args = (c, pairs, widths)
    elif op == "joint_dense_counts":
        args = (c, pairs, widths, 4)
    else:
        keys = fidelity_ops.joint_keys(c, pairs, widths).sort(dim=1).values
        n = torch.tensor([3], device=device)
        args = (keys, keys, n, n, x.new_empty(()))
    assert all(
        v == "SUCCESS"
        for v in torch.library.opcheck(
            getattr(fidelity_ops, op), args, test_utils=("test_schema", "test_faketensor")
        ).values()
    )


@pytest.mark.parametrize("B", [0, 2, 5])
def test_full_trend_reports_preserve_replicates_flags_and_pairing(torch, device, dtype, B):
    r, s, schema = frames(n=260)
    common = dict(
        dtype=dtype,
        uncertainty="both" if B else "none",
        n_replicates=max(B, 1),
        seed=62,
    )
    expected = evaluate(
        r, s, schema, metrics=["trend"], options=make_options(device="cpu", **common)
    )
    actual = evaluate(
        r, s, schema, metrics=["trend"], options=make_options(device=device, **common)
    )
    x, y = expected["trend/trend"], actual["trend/trend"]
    assert x.flags == y.flags
    for field in [
        "value",
        "ideal",
        "ci_low",
        "ci_high",
        "ideal_ci_low",
        "ideal_ci_high",
        "generated_distribution",
        "reference_distribution",
    ]:
        a, b = getattr(x, field), getattr(y, field)

        if hasattr(a, "samples"):
            a, b = a.samples, b.samples
        if a is None:
            assert b is None
        else:
            np.testing.assert_allclose(
                a, b, rtol=0, atol=3e-6 if dtype == "float32" else 1e-12, equal_nan=True
            )
    info = actual.provenance["metric_execution"]["trend"]
    assert info["implementation"] == ("triton" if device.startswith("cuda") else "aten_cpp")
    assert info["device"] == device and info["dtype"] == dtype


def test_trend_preserves_cells_duplicate_draws_and_input_ownership(torch, device, dtype):
    r, s, schema = frames()
    r["ignored"] = np.nan
    s["ignored"] = np.nan
    metric = Trend(device=device, dtype=dtype)
    context = context_for(metric, r, s, schema)
    original = context.reference.numeric.clone()
    rc = np.array([[0, 2, 2, 7, 11, 19, 19, 22, 31, 55]])
    sc = np.array([[0, 3, 3, 9, 12, 20, 20, 23, 32, 54]])
    expected = Trend().compute(r.iloc[rc[0]], s.iloc[sc[0]], schema, rng=None)["trend"]
    got = context.compute(metric, rc, sc)["trend"][0]
    assert got == pytest.approx(expected, abs=3e-6 if dtype == "float32" else 1e-12)
    torch.testing.assert_close(context.reference.numeric, original, rtol=0, atol=0, equal_nan=True)
    assert context.compute(metric, rc[:0], sc[:0])["trend"].shape == (0,)
    for bad in [np.array([[-1]]), np.array([[len(r)]]), np.array([[1.5]])]:
        with pytest.raises(ValueError, match="draw"):
            context.compute(metric, bad, sc[:, :1])
    with pytest.raises(ValueError, match="batch lengths"):
        context.compute(metric, rc, sc[:0])


def test_trend_cpu_has_no_triton_dependency(torch, monkeypatch, dtype):
    from tabgen_eval.execution.torch.fidelity import ops as fidelity_ops

    def forbidden(*args, **kwargs):
        pytest.fail("tensor CPU Trend must not import Triton")

    monkeypatch.setattr(fidelity_ops, "cuda_kernels", forbidden)
    r, s, schema = frames()
    Trend(device="cpu", dtype=dtype).compute(r, s, schema, rng=None)


def test_actual_none_is_missing_and_literal_nan_is_observed(torch, device, dtype):
    r = pd.DataFrame({"a": [None, "nan", "x", "y"], "b": ["u", "u", "u", "v"]})
    s = pd.DataFrame({"a": [None, "nan", "x", "x"], "b": ["u", "u", "u", "v"]})
    # Three observed cells each carry mass 1/3; two overlap the candidate.
    expected = 2 / 3
    actual = Trend(device=device, dtype=dtype).compute(
        r, s, Schema(categorical=["a", "b"]), rng=None
    )["trend"]
    assert actual == pytest.approx(expected, abs=2e-7 if dtype == "float32" else 1e-15)


@pytest.mark.parametrize("B", [0, 2, 5])
def test_tensor_trend_matches_every_native_report_field(torch, device, B):
    r, s, schema = frames(n=260)
    common = dict(uncertainty="both" if B else "none", n_replicates=max(B, 1), n_jobs=4, seed=718)
    a = evaluate(r, s, schema, metrics=["trend"], options=make_options(**common))["trend/trend"]
    b = evaluate(
        r,
        s,
        schema,
        metrics=["trend"],
        options=make_options(device=device, dtype="float64", **common),
    )["trend/trend"]
    assert a.flags == b.flags
    for field in [
        "value",
        "ideal",
        "ci_low",
        "ci_high",
        "ideal_ci_low",
        "ideal_ci_high",
        "generated_distribution",
        "reference_distribution",
    ]:
        x, y = getattr(a, field), getattr(b, field)

        if hasattr(x, "samples"):
            x, y = x.samples, y.samples
        if x is None:
            assert y is None
        else:
            np.testing.assert_allclose(x, y, atol=1e-12, rtol=0, equal_nan=True)


@pytest.mark.gpu
def test_trend_gpu_workspace_multiple_draws_and_sparse_budget_independence(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    r, s, schema = frames(n=800, cardinality=15000)
    a = Trend(device=device, gpu_workspace_mb=2)
    context = context_for(a, r, s, schema)
    rd, sd = np.arange(len(r))[None], np.arange(len(s))[None]
    context.compute(a, rd, sd)
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = context.compute(a, np.repeat(rd, 4, axis=0), np.repeat(sd, 4, axis=0))["trend"]
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) - base <= 2 * 1024**2
    b = Trend(device=device, gpu_workspace_mb=32)
    expected = context_for(b, r, s, schema).compute(b, rd, sd)["trend"][0]
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)
    with pytest.raises(MemoryError, match="workspace allowance"):
        context.compute(
            a, np.zeros((1, 100000), dtype=np.int64), np.zeros((1, 100000), dtype=np.int64)
        )


@pytest.mark.gpu
def test_trend_respects_selected_precision_stream_and_device(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from tabgen_eval.execution.torch.fidelity.ops import fidelity_gram

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    rng = np.random.default_rng(187)
    x = torch.tensor(rng.normal(size=(4199, 23)), dtype=torch.float32, device=device)
    expected = x.cpu().double().T @ x.cpu().double()
    precision, current = torch.get_float32_matmul_precision(), torch.cuda.current_device()
    stream = torch.cuda.Stream(device=device)
    try:
        # Deliberately permit TF32 for ordinary PyTorch matmuls; the registered
        # operator must still use IEEE FP32 and leave the caller's setting alone.
        torch.set_float32_matmul_precision("high")
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16):
            got = fidelity_gram(x, x)
        stream.synchronize()
        assert got.dtype == torch.float32
        np.testing.assert_allclose(got.cpu().numpy(), expected.numpy(), atol=0.003, rtol=1e-5)
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.cuda.current_device() == current
    finally:
        torch.set_float32_matmul_precision(precision)


def test_typed_category_families_have_independent_joint_mass(torch, device, dtype):
    # Equal-valued Python scalars have different categorical identities. Every
    # row has reference mass 1/4; swapping two labels removes exactly half of
    # the shared joint mass, independent of code ordering or numerical precision.
    reference = pd.DataFrame(
        {"a": pd.Series([1, 1.0, "1", True], dtype=object), "b": ["i", "f", "s", "b"]}
    )
    candidate = reference.copy(deep=True)
    candidate["a"] = pd.Series([1.0, 1, "1", True], dtype=object)
    schema = Schema(categorical=["a", "b"])
    actual = Trend(device=device, dtype=dtype).compute(reference, candidate, schema, rng=None)[
        "trend"
    ]
    assert actual == 0.5
