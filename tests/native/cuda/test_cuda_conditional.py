"""Grouped Conditional GPU math and complete estimator/report contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import Conditional
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "kind,cardinality",
    [("numeric", 10)] + [(kind, n) for kind in ["categorical", "mixed"] for n in [10, 1024, 15000]],
)
def test_conditional_all_statistics_against_cpu(torch, device, dtype, kind, cardinality):
    rng = np.random.default_rng(1221)
    numeric = ["x", "y", "z"] if kind != "categorical" else []
    categorical = ["a", "b", "c"] if kind != "numeric" else []
    frames = []
    for n in [263, 247]:
        frame = pd.DataFrame({c: rng.normal(size=n).round(1).astype(dtype) for c in numeric})
        for c in categorical:
            frame[c] = rng.integers(cardinality, size=n).astype(str)
            frame.loc[::29, c] = "nan"
            frame.loc[::31, c] = None
        for c in numeric:
            frame.loc[::37, c] = np.nan
        if numeric:
            frame.loc[::23, numeric[0]] = np.nan
        frames.append(frame)
    schema = Schema(numeric=numeric, categorical=categorical)
    options = dict(min_count=2, rare_threshold=0.15, num_bins=5)
    expected = Conditional(device="cpu", dtype=dtype, **options).compute(*frames, schema, rng=None)
    actual = Conditional(device=device, dtype=dtype, **options).compute(*frames, schema, rng=None)
    np.testing.assert_allclose(
        list(actual.values()),
        list(expected.values()),
        rtol=0,
        atol=4e-6 if dtype == "float32" else 1e-12,
        equal_nan=True,
    )


def test_count_and_rarity_boundaries_empty_candidate_and_literal_nan(torch, device):
    real = pd.DataFrame(
        {
            "x": [7.0] * 100,
            "a": ["nine"] * 9 + ["ten"] * 10 + ["eleven"] * 11 + ["nan"] * 70,
            "b": ["b", "c"] * 50,
        }
    )
    generated = real.loc[real.a != "ten"].copy()
    schema = Schema(numeric=["x"], categorical=["a", "b"])
    options = dict(min_count=10, rare_threshold=0.1)
    metric = Conditional(device=device, **options)
    context = context_for(metric, real, generated, schema)
    assert context.not_applicable(metric, schema) == {
        "cond_shape_rare",
        "cond_trend_rare",
        "recovery_rare",
    }
    expected = Conditional(**options).compute(real, generated, schema, rng=None)
    actual = metric.compute(real, generated, schema, rng=None)
    np.testing.assert_allclose(
        list(actual.values()), list(expected.values()), atol=1e-12, rtol=0, equal_nan=True
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparse_global_domains_repeated_unequal_draws_and_ideal(torch, device, dtype):
    rng = np.random.default_rng(1222)
    numeric = ["x", "y"]
    schema = Schema(numeric=numeric, categorical=["a", "b"])
    real = pd.DataFrame(rng.normal(size=(6000, 2)).astype(dtype), columns=numeric)
    generated = pd.DataFrame(rng.normal(size=(6000, 2)).astype(dtype), columns=numeric)
    for frame in [real, generated]:
        frame["a"] = np.arange(6000).astype(str)
        frame["b"] = np.arange(6000)[::-1].astype(str)
        frame.loc[::17, "a"] = None
        frame.loc[::19, "x"] = np.nan
    rd = rng.integers(0, 70, size=(2, 99), dtype=np.int64)
    sd = rng.integers(0, 90, size=(2, 113), dtype=np.int64)
    metric = Conditional(device=device, dtype=dtype, min_count=1)
    context = context_for(metric, real, generated, schema)
    for ideal in [False, True]:
        result = context.compute(metric, rd, sd, ideal=ideal)
        for i in range(2):
            expected = Conditional(min_count=1).compute(
                real.iloc[rd[i]], (real if ideal else generated).iloc[sd[i]], schema, rng=None
            )
            np.testing.assert_allclose(
                [result[k][i] for k in metric.stat_names],
                list(expected.values()),
                rtol=0,
                atol=4e-6 if dtype == "float32" else 1e-12,
                equal_nan=True,
            )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("bootstrap", [0, 2])
def test_complete_conditional_report_and_precision_isolation(torch, device, dtype, bootstrap):
    rng = np.random.default_rng(1223)
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    real = pd.DataFrame(
        {
            "x": rng.normal(size=300).astype(dtype),
            "y": rng.normal(size=300).astype(dtype),
            "c": rng.choice(["a", "b", "rare"], 300, p=[0.5, 0.46, 0.04]),
        }
    )
    generated = real.copy()
    generated.loc[::31, "x"] = np.nan
    before = (
        torch.cuda.current_device(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            actual = evaluate(
                real,
                generated,
                schema,
                metrics=["conditional_fidelity"],
                options=make_options(
                    device=device,
                    dtype=dtype,
                    uncertainty="both" if bootstrap else "none",
                    n_replicates=max(1, bootstrap),
                ),
            )
        expected = evaluate(
            real,
            generated,
            schema,
            metrics=["conditional_fidelity"],
            options=make_options(
                uncertainty="both" if bootstrap else "none",
                n_replicates=max(1, bootstrap),
            ),
        )
        for key, stat in actual.results[0].stats.items():
            other = expected.results[0].stats[key]
            assert stat.flags == other.flags
            np.testing.assert_allclose(
                stat.value,
                other.value,
                rtol=0,
                atol=4e-6 if dtype == "float32" else 1e-12,
                equal_nan=True,
            )
            for field in ["generated_distribution", "reference_distribution"]:
                if getattr(stat, field) is not None:
                    np.testing.assert_allclose(
                        (
                            getattr(stat, field).samples
                            if field.endswith("_distribution")
                            else getattr(stat, field)
                        ),
                        (
                            getattr(other, field).samples
                            if field.endswith("_distribution")
                            else getattr(other, field)
                        ),
                        rtol=0,
                        atol=4e-6 if dtype == "float32" else 1e-12,
                        equal_nan=True,
                    )
        assert (
            actual.provenance["metric_execution"]["conditional_fidelity"]["implementation"]
            == "triton"
        )
    finally:
        torch.set_float32_matmul_precision(before[1])
        torch.backends.cuda.matmul.allow_tf32 = before[2]
    assert torch.cuda.current_device() == before[0]


def test_workspace_is_bounded_and_draws_are_released(torch, device):
    rng = np.random.default_rng(1224)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=240),
            "y": rng.normal(size=240),
            "c": rng.integers(4, size=240).astype(str),
        }
    )
    metric = Conditional(device=device, gpu_workspace_mb=1)
    context = context_for(metric, frame, frame, Schema(numeric=["x", "y"], categorical=["c"]))
    draw = np.arange(180, dtype=np.int64)[None]
    context.compute(metric, draw, draw)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    actual = context.compute(metric, np.repeat(draw, 3, axis=0), np.repeat(draw, 3, axis=0))
    assert torch.cuda.max_memory_allocated(device) - baseline <= 1024**2
    np.testing.assert_array_equal(actual["cond_shape_all"], np.ones(3))
    assert torch.cuda.memory_allocated(device) == baseline
    huge = np.zeros((1, 100000), dtype=np.int64)
    with pytest.raises(MemoryError, match="workspace allowance"):
        context.compute(metric, huge, huge)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_generated_only_levels_preserve_all_scores_and_draw_local_support(torch, device, dtype):
    rng = np.random.default_rng(1225)
    real = pd.DataFrame(
        {
            "x": rng.normal(size=180).astype(dtype),
            "slice": ["a"] * 60 + ["b"] * 120,
            "wide": ["known_a", "known_b"] * 90,
            "special": ["nan", "ok", None] * 60,
        }
    )
    generated = pd.DataFrame(
        {
            "x": rng.normal(size=240).astype(dtype),
            "slice": ["a"] * 80 + ["b"] * 160,
            "wide": ["known_a", "known_b"] * 30 + [f"novel_{i}" for i in range(180)],
            "special": ["nan", "ok", "novel", None] * 60,
        }
    )
    real.loc[::17, "wide"] = None
    generated.loc[::19, "wide"] = None
    schema = Schema(numeric=["x"], categorical=["slice", "wide", "special"])
    options = dict(rare_threshold=0.4, min_count=5)
    metric = Conditional(device=device, dtype=dtype, **options)
    context = context_for(metric, real, generated, schema)
    # The second draw removes one reference level: mappings must be refitted.
    rd = np.stack([np.arange(180), np.tile(np.arange(0, 180, 2), 2)])
    sd = np.stack([np.arange(180), np.arange(60, 240)])
    original_real = context.reference.categorical.clone()
    original_generated = context.candidate.categorical.clone()
    for ideal in [False, True]:
        candidate_draws = rd if ideal else sd
        actual = context.compute(metric, rd, candidate_draws, ideal=ideal)
        for i in range(2):
            expected = Conditional(device="cpu", dtype=dtype, **options).compute(
                real.iloc[rd[i]],
                (real if ideal else generated).iloc[candidate_draws[i]],
                schema,
                rng=None,
            )
            np.testing.assert_allclose(
                [actual[name][i] for name in metric.stat_names],
                list(expected.values()),
                rtol=0,
                atol=4e-6 if dtype == "float32" else 1e-12,
                equal_nan=True,
            )
    torch.testing.assert_close(context.reference.categorical, original_real)
    torch.testing.assert_close(context.candidate.categorical, original_generated)


def test_histogram_runtime_pair_count_preserves_group_strides(torch, device):
    from tabgen_eval.execution.torch.fidelity.grouped_ops import group_dense_counts

    rng = np.random.default_rng(1226)
    source = rng.integers(-1, 4, size=(3, 4, 128), dtype=np.int64)
    codes = torch.tensor(source, device=device)
    all_pairs = [(i, j) for i in range(4) for j in range(4)] * 2
    for q in [1, 2, 5, 16, 17]:
        pairs = torch.tensor(all_pairs[:q], device=device, dtype=torch.int64)
        widths = torch.full((3, q), 4, device=device, dtype=torch.int64)
        actual = group_dense_counts(codes, pairs, widths, 16).cpu().numpy()
        expected = np.zeros((3, q, 16), dtype=np.int64)
        for group in range(3):
            for pair, (i, j) in enumerate(all_pairs[:q]):
                a, b = source[group, i], source[group, j]
                valid = (a >= 0) & (b >= 0)
                expected[group, pair] = np.bincount(a[valid] * 4 + b[valid], minlength=16)
        np.testing.assert_array_equal(actual, expected)


def test_small_vocabulary_skips_compaction_work(torch, device, monkeypatch):
    from tabgen_eval.execution.torch.fidelity import conditional

    real = pd.DataFrame({"x": np.arange(80.0), "a": ["a", "b"] * 40, "b": ["c", "d"] * 40})
    schema = Schema(numeric=["x"], categorical=["a", "b"])
    metric = Conditional(device=device)
    context = context_for(metric, real, real, schema)
    assert context.reference_cardinalities == (2, 2)

    def unexpected_compaction(*args, **kwargs):
        raise AssertionError("Small shared vocabularies should retain the original code tables")

    monkeypatch.setattr(conditional, "compact_categories", unexpected_compaction)
    draws = np.arange(80)[None]
    actual = context.compute(metric, draws, draws)
    assert actual["cond_shape_all"][0] == 1
    assert actual["recovery_all"][0] == 1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("marginal", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_fused_tv_against_independent_histograms(torch, device, dtype, marginal, native):
    from tabgen_eval.execution.torch.cuda.fidelity.native_tv import dense_tv_batches
    from tabgen_eval.execution.torch.fidelity.grouped_ops import group_dense_tv

    rng = np.random.default_rng(1227)
    a = rng.integers(-1, 4, size=(3, 4, 67), dtype=np.int64)
    b = rng.integers(-1, 4, size=(3, 4, 53), dtype=np.int64)
    a[1, 2], b[1, 2], b[2] = -1, -1, -1
    levels = np.full((3, 4), 4, dtype=np.int64)
    levels[1, 2] = 0
    pairs = [(i, i) for i in range(4)] if marginal else [(0, 1), (1, 2), (0, 3), (2, 3)]
    grid = 4 if marginal else 16
    args = [torch.tensor(x, device=device) for x in (a, b, np.array(pairs), levels)]
    prototype = torch.empty((), dtype=getattr(torch, dtype), device=device)
    for cond in [-1, 0, 2]:
        result = (
            torch.cat(dense_tv_batches(*args, grid, 3, cond, marginal, prototype), dim=1)
            if native
            else group_dense_tv(*args, grid, cond, marginal, prototype)
        )
        actual = result.cpu().numpy()
        expected = np.full((3, len(pairs)), np.nan)
        for g in range(3):
            for q, (i, j) in enumerate(pairs):
                if cond in (i, j):
                    continue
                counts = []
                for source in (a, b):
                    keep = (source[g, i] >= 0) & (source[g, j] >= 0)
                    keys = (
                        source[g, j, keep]
                        if marginal
                        else source[g, i, keep] * 4 + source[g, j, keep]
                    )
                    counts.append(np.bincount(keys, minlength=grid))
                if counts[0].sum() and counts[1].sum():
                    expected[g, q] = (
                        1
                        - 0.5
                        * np.abs(counts[0] / counts[0].sum() - counts[1] / counts[1].sum()).sum()
                    )
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=4e-6 if dtype == "float32" else 1e-12, equal_nan=True
        )
    np.testing.assert_array_equal(args[0].cpu(), a)
    np.testing.assert_array_equal(args[1].cpu(), b)


def test_native_tv_current_stream_device_and_empty_outputs(torch, device):
    from tabgen_eval.execution.torch.cuda.fidelity.native_tv import _compiled, dense_tv_batches
    from tabgen_eval.execution.torch.fidelity.grouped_ops import group_dense_tv

    # Compile outside the nondefault stream, then enqueue inputs and native
    # launches together. CPU waiting uses that stream's event, not device sync.
    target = torch.device(device)
    _compiled(target, 16, False, torch.float64)
    stream = torch.cuda.Stream(device=target)
    previous = torch.cuda.current_device()
    with torch.cuda.stream(stream):
        a = torch.randint(-1, 4, (2, 3, 2051), device=target)
        b = torch.randint(-1, 4, (2, 3, 1027), device=target)
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], device=target)
        levels = torch.full((2, 3), 4, device=target)
        prototype = torch.empty((), device=target, dtype=torch.float64)
        expected = group_dense_tv(a, b, pairs, levels, 16, -1, False, prototype)
        outputs = dense_tv_batches(a, b, pairs, levels, 16, 2, -1, False, prototype)
        actual = torch.cat(outputs, dim=1)
        event = stream.record_event()
    event.synchronize()
    assert torch.cuda.current_device() == previous
    assert [tuple(x.shape) for x in outputs] == [(2, 2), (2, 1)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    empty = dense_tv_batches(a, b, pairs[:0], levels, 16, 2, -1, False, prototype)
    assert empty[0].shape == (2, 0)


def test_native_tv_fake_contract_and_launch_bounds(torch, device):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tabgen_eval.execution.torch.cuda.extension import library
    from tabgen_eval.execution.torch.cuda.fidelity.native_tv import _compiled

    native = library()
    _, _, hist, tv = _compiled(torch.device(device), 16, False, torch.float64)
    a = torch.zeros((2, 3, 5), device=device, dtype=torch.int64)
    pairs = torch.zeros((3, 2), device=device, dtype=torch.int64)
    levels = torch.ones((2, 3), device=device, dtype=torch.int64)
    prototype = torch.empty((), device=device, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="launch bounds"):
        native.dense_tv_batches(a, a, pairs, levels, prototype, 16, 0, -1, hist, tv)
    with pytest.raises(RuntimeError, match="launch metadata"):
        native.dense_tv_batches(a, a, pairs, levels, prototype, 16, 2, -1, [], tv)
    offset = torch.zeros((31,), device=device, dtype=torch.int64)[1:].view_as(a)
    with pytest.raises(RuntimeError, match="aligned inputs"):
        native.dense_tv_batches(offset, a, pairs, levels, prototype, 16, 2, -1, hist, tv)
    with FakeTensorMode() as mode:
        args = [mode.from_tensor(x) for x in (a, a, pairs, levels, prototype)]
        outputs = native.dense_tv_batches(*args, 16, 2, -1, hist, tv)
        assert [tuple(x.shape) for x in outputs] == [(2, 2), (2, 1)]
        assert all(x.device == prototype.device and x.dtype == prototype.dtype for x in outputs)


@pytest.mark.parametrize(
    "unsupported",
    [
        "abi",
        "num_ctas",
        "launch_cooperative_grid",
        "launch_pdl",
        "global_scratch_size",
        "profile_scratch_size",
    ],
)
def test_native_tv_rejects_unsupported_compiler_metadata(unsupported):
    from types import SimpleNamespace

    from tabgen_eval.execution.torch.cuda.fidelity.native_tv import _spec

    metadata = SimpleNamespace(num_ctas=1)
    signature = {"x": "*i64" if unsupported != "abi" else "i32"}
    if unsupported != "abi":
        setattr(metadata, unsupported, 2)
    kernel = SimpleNamespace(
        src=SimpleNamespace(signature=signature),
        metadata=metadata,
        _init_handles=lambda: pytest.fail("unsupported kernel must not be loaded"),
    )
    with pytest.raises(RuntimeError, match="unsupported native Conditional kernel"):
        _spec(kernel, ["*i64"])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fused_center_preserves_extremes_and_overlap_masks(torch, device, dtype):
    from tabgen_eval.execution.torch.fidelity.grouped_ops import group_center
    from tabgen_eval.execution.torch.fidelity.primitives import centered_columns

    rng = np.random.default_rng(1228)
    data = rng.normal(size=(2, 8, 2057)).astype(dtype)
    largest = np.finfo(dtype).max / 2
    data[:, 0] = largest * np.resize(np.array([-1.5, 1.5, 0.1]), 2057)
    offset = np.array(1e20, dtype=dtype)
    data[:, 1] = offset + np.spacing(offset) * rng.integers(-4, 5, size=(2, 2057))
    data[:, 2] = 7
    data[:, 3] = np.nan
    data[:, 4, ::19] = np.inf
    data[:, 6] = np.finfo(dtype).tiny / 128 * rng.integers(-4, 5, size=(2, 2057))
    data[:, 7] = np.nan
    data[0, 7, 1030:] = rng.normal(size=1027)  # first anchor is in a later tile
    valid = np.isfinite(data)
    valid[:, 5, ::3] = False  # finite cells excluded by pairwise overlap
    values = torch.tensor(data, device=device)
    mask = torch.tensor(valid, device=device)
    actual, variable = group_center(values, mask)
    expected, varies = [], []
    for group, keep in zip(data, valid, strict=True):
        centered, flags = centered_columns(torch.tensor(group.T), torch.tensor(keep.T))
        expected.append(centered.T.numpy())
        varies.append(flags.numpy())
    np.testing.assert_allclose(
        actual.cpu(), np.array(expected), rtol=0, atol=4e-6 if dtype == "float32" else 1e-12
    )
    np.testing.assert_array_equal(variable.cpu(), np.array(varies))
    torch.testing.assert_close(values, torch.tensor(data, device=device), equal_nan=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("bins", [1, 5, 10, 65])
def test_fused_edges_preserve_ties_constants_and_empty_columns(torch, device, dtype, bins):
    from tabgen_eval.execution.torch.fidelity.grouped_ops import group_edges
    from tabgen_eval.execution.torch.fidelity.primitives import quantile_edges

    data = np.array(
        [[7] * 17, [0] * 6 + [1] * 6 + [2] * 5, np.linspace(-3, 3, 17), [np.nan] * 17], dtype=dtype
    )
    counts = np.isfinite(data).sum(axis=1)
    ordered = np.sort(np.where(np.isfinite(data), data, np.inf), axis=1)[None]
    values = torch.tensor(ordered, device=device)
    sizes = torch.tensor(counts[None], device=device)
    actual, levels = group_edges(values, sizes, bins)
    for column, source in enumerate(data):
        expected = quantile_edges(torch.tensor(source), bins)
        if expected is None:
            assert levels[0, column].item() == 0
            continue
        assert levels[0, column].item() == len(expected) - 1
        np.testing.assert_allclose(
            actual[0, column, : len(expected)].cpu(),
            expected,
            rtol=0,
            atol=4e-6 if dtype == "float32" else 1e-12,
        )
        assert torch.isinf(actual[0, column, len(expected) :]).all()


def test_fused_group_operators_fake_contracts(torch, device):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tabgen_eval.execution.torch.fidelity.grouped_ops import (
        group_center,
        group_dense_tv,
        group_edges,
    )

    with FakeTensorMode():
        values = torch.empty((3, 4, 128), device=device, dtype=torch.float64)
        valid = torch.empty_like(values, dtype=torch.bool)
        counts = torch.empty((3, 4), device=device, dtype=torch.int64)
        centered, variable = group_center(values, valid)
        edges, levels = group_edges(values, counts, 10)
        pairs = torch.empty((5, 2), device=device, dtype=torch.int64)
        scores = group_dense_tv(
            torch.empty_like(values, dtype=torch.int64),
            torch.empty((3, 4, 64), device=device, dtype=torch.int64),
            pairs,
            levels,
            16,
            0,
            False,
            values.new_empty(()),
        )
        assert centered.shape == values.shape and centered.dtype == torch.float64
        assert variable.shape == counts.shape and variable.dtype == torch.bool
        assert edges.shape == (3, 4, 11) and levels.dtype == torch.int64
        assert scores.shape == (3, 5) and scores.dtype == torch.float64
