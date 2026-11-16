"""CUDA detail tables retain point reductions, source identities and fitted bins."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.execution.python.primitives import (
    contingency_similarity,
    correlation_similarity,
    ks_complement,
    tv_complement,
)
from tabgen_eval.execution.torch.context import TorchContext
from tabgen_eval.metrics import Conditional, Shape, Trend
from tabgen_eval.preprocessing.columns import PreparedColumns
from tabgen_eval.schema import Schema

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def context_for_details(real, candidate, schema, device, dtype):
    columns = PreparedColumns.prepare(real, candidate, schema, retain_labels=True)
    return TorchContext.from_columns(columns, device=device, dtype=dtype)


def mixed_frames(dtype):
    rng = np.random.default_rng(9441)
    frames = []
    for n in [263, 247]:
        frame = pd.DataFrame(
            dict(x=rng.normal(size=n).astype(dtype), y=rng.normal(size=n).astype(dtype))
        )
        for name, cardinality in [("a", 2), ("b", 7), ("c", 70)]:
            frame[name] = rng.integers(cardinality, size=n).astype(str)
        frame["missing"] = None
        frame.loc[::17, "x"] = np.nan
        frame.loc[::19, "b"] = None
        frames.append(frame)
    return *frames, Schema(numeric=["x", "y"], categorical=["a", "b", "c", "missing"])


@pytest.mark.parametrize("metric_type", [Shape, Trend, Conditional])
@pytest.mark.parametrize("ideal", [False, True])
def test_detail_capture_preserves_scalar_bits(torch, device, dtype, metric_type, ideal):
    real, candidate, schema = mixed_frames(dtype)
    context = context_for_details(real, candidate, schema, device, dtype)
    options = (
        dict(min_count=2, num_bins=5, rare_threshold=0.1) if metric_type is Conditional else {}
    )
    metric = metric_type(**options)
    rng = np.random.default_rng(9442)
    rd = rng.integers(0, len(real), 141)
    sd = rng.integers(0, len(real) if ideal else len(candidate), 133)
    plain = context.compute(metric, rd[None, :], sd[None, :], ideal=ideal)
    detailed = context.compute_detailed(metric, rd, sd, ideal=ideal)
    for name, values in plain.items():
        if np.isnan(values[0]):
            assert np.isnan(detailed.values[name])
        else:
            assert float(values[0]).hex() == float(detailed.values[name]).hex()


def test_global_column_and_bucketed_pair_identities(torch, device, dtype):
    real, candidate, schema = mixed_frames(dtype)
    for frame in (real, candidate):
        frame["sparse"] = (np.arange(len(frame)) % 71).astype(str)
        frame["empty_numeric"] = np.nan
    schema.categorical.append("sparse")
    schema.numeric.append("empty_numeric")
    context = context_for_details(real, candidate, schema, device, dtype)
    assert context.inputs.cardinalities[2] * context.inputs.cardinalities[-1] > 4096
    rd, sd = np.arange(len(real)), np.arange(len(candidate))
    shape = context.compute_detailed(Shape(), rd, sd).details.tables["columns"].columns
    expected = [ks_complement(real[c], candidate[c]) for c in schema.numeric]
    expected += [tv_complement(real[c], candidate[c]) for c in schema.categorical]
    np.testing.assert_allclose(shape["score"], expected, rtol=0, atol=2e-6, equal_nan=True)
    trend = context.compute_detailed(Trend(), rd, sd).details.tables["pairs"].columns
    columns = schema.numeric + schema.categorical
    pairs = list(combinations(range(len(columns)), 2))
    assert list(zip(trend["column_1_index"], trend["column_2_index"], strict=True)) == pairs
    expected = []
    for i, j in pairs:
        a, b = columns[i], columns[j]
        if j < len(schema.numeric):
            score = correlation_similarity(real[a], real[b], candidate[a], candidate[b])
        else:
            score = contingency_similarity(
                real[a],
                real[b],
                candidate[a],
                candidate[b],
                col1_numeric=i < len(schema.numeric),
                col2_numeric=j < len(schema.numeric),
            )
        expected.append(score)
    np.testing.assert_allclose(trend["score"], expected, rtol=0, atol=3e-6, equal_nan=True)


def test_conditional_group_ids_counts_exclusions_and_draw_local_bounds(torch, device, dtype):
    rng = np.random.default_rng(9443)
    labels = ["nan", " A ", "A", "<MISSING>", "five", "six", "seven"]
    counts = [3, 18, 5, 40, 2, 25, 11]
    ref_labels = np.repeat(labels, counts)
    real = pd.DataFrame(
        dict(
            x=rng.normal(size=len(ref_labels) + 10).astype(dtype),
            y=rng.normal(size=len(ref_labels) + 10).astype(dtype),
            c=["undrawn"] * 10 + ref_labels.tolist(),
        )
    )
    real.loc[:9, "x"] = 1e6
    # Generated-only identities trigger draw-local TV compaction; conditioning
    # identities must still use the unmodified shared vocabulary.
    gen_labels = ["new_" + str(i) for i in range(100)] + [" A "] * 12 + ["six"] * 7
    candidate = pd.DataFrame(
        dict(
            x=rng.normal(size=len(gen_labels)).astype(dtype),
            y=rng.normal(size=len(gen_labels)).astype(dtype),
            c=gen_labels,
        )
    )
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    context = context_for_details(real, candidate, schema, device, dtype)
    rd, sd = np.arange(10, len(real)), np.arange(len(candidate))
    metric = Conditional(min_count=6, num_bins=4, rare_threshold=0.1)
    result = context.compute_detailed(metric, rd, sd)
    table = result.details.tables["slices"].columns
    selected = table["column_index"] == 2
    rows = {int(code): index for index, code in enumerate(table["label_code"]) if selected[index]}
    reference_codes = context.reference.categorical[rd, 0].cpu().numpy()
    candidate_codes = context.candidate.categorical[sd, 0].cpu().numpy()
    assert list(rows) == sorted(np.unique(reference_codes).tolist())
    assert min(rows) > 0  # the undrawn source identity occupied code zero
    for code, index in rows.items():
        rc, sc = int((reference_codes == code).sum()), int((candidate_codes == code).sum())
        assert table["reference_count"][index] == rc
        assert table["candidate_count"][index] == sc
        assert table["reference_total"][index] == len(rd)
        assert table["candidate_total"][index] == len(sd)
        assert table["eligible"][index] == (rc >= 6)
        assert table["rare"][index] == (rc / len(rd) < 0.1)
        if rc < 6:
            assert np.isnan(table["shape"][index])
            assert np.isnan(table["trend"][index])
            assert np.isnan(table["recovery"][index])
        elif not sc:
            assert table["shape"][index] == table["trend"][index] == table["recovery"][index] == 0
        else:
            assert table["recovery"][index] == pytest.approx(
                (sc / len(sd)) / (rc / len(rd)), rel=1e-6
            )
        assert np.isnan(table["lower"][index]) and np.isnan(table["upper"][index])
        assert not table["left_closed"][index] and not table["right_closed"][index]
    numeric = table["column_index"] == 0
    assert table["upper"][numeric].max() < 10
    assert table["lower"][numeric].min() == pytest.approx(real.iloc[rd].x.min())
    assert table["upper"][numeric].max() == pytest.approx(real.iloc[rd].x.max())
    assert table["right_closed"][numeric].all()
    np.testing.assert_array_equal(table["left_closed"][numeric], table["label_code"][numeric] == 0)


@pytest.mark.parametrize("width", [1, 2])
def test_structurally_unavailable_detail_scores_stay_nan(torch, device, dtype, width):
    real = pd.DataFrame(dict(c=["a"] * 12 + ["b"] * 3))
    candidate = pd.DataFrame(dict(c=["b"] * 15))
    schema = Schema(categorical=["c"])
    if width == 2:
        real["x"], candidate["x"] = np.arange(15).astype(dtype), np.arange(15).astype(dtype)
        schema.numeric = ["x"]
    context = context_for_details(real, candidate, schema, device, dtype)
    metric = Conditional(min_count=2)
    result = context.compute_detailed(metric, np.arange(15), np.arange(15))
    slices = result.details.tables["slices"].columns
    columns = result.details.tables["columns"].columns
    assert np.isnan(slices["trend"]).all()
    assert np.isnan(columns["cond_trend_all"]).all()
    assert np.isnan(columns["cond_trend_rare"]).all()
    if width == 1:
        assert all(np.isnan(slices[name]).all() for name in ["shape", "trend"])
        assert all(
            np.isnan(columns[name]).all()
            for name in metric.stat_names
            if not name.startswith("recovery")
        )
        np.testing.assert_allclose(slices["recovery"], [0.0, 5.0])
        np.testing.assert_allclose(columns["recovery_all"], [2.5])


def test_conditional_capture_does_not_repeat_score_batches(torch, device, monkeypatch):
    from tabgen_eval.execution.torch.fidelity.conditional import ConditionalExecutor

    real, candidate, schema = mixed_frames("float64")
    context = context_for_details(real, candidate, schema, device, "float64")
    executor = ConditionalExecutor(context)
    metric = Conditional(min_count=2, num_bins=5)
    original = executor._scores
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "_scores", wrapped)
    rd, sd = np.arange(len(real)), np.arange(len(candidate))
    executor.compute(metric, rd[None, :], sd[None, :])
    count = len(calls)
    calls.clear()
    executor.compute_detailed(metric, rd, sd)
    assert len(calls) == count > 0


def test_disabled_capture_does_not_construct_detail_collector(torch, device, monkeypatch):
    from tabgen_eval.execution.torch.fidelity import conditional

    real, candidate, schema = mixed_frames("float64")
    context = context_for_details(real, candidate, schema, device, "float64")

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled capture must not allocate a detail collector")

    monkeypatch.setattr(conditional, "ConditionalDetails", unexpected)
    context.compute(
        Conditional(min_count=2), np.arange(len(real))[None, :], np.arange(len(candidate))[None, :]
    )


def test_host_transfer_is_owned_and_chunks_existing_buffers(torch):
    from tabgen_eval.execution.torch.fidelity.details import host_array

    source = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    full = host_array(source)
    chunked = host_array(source, max_bytes=32)
    source.fill_(-1)
    np.testing.assert_array_equal(full, np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(chunked, full)
