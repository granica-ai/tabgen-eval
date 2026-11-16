"""Independent holdout baselines against explicit training-fitted distance oracles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.execution.runtime.outcomes import MemorizationOutcome
from tabgen_eval.metrics import DCR
from tabgen_eval.schema.validate import ValidationError
from tests.support.distances import distance_matrices
from tests.support.distances import memorization as compute
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def dtype(backend_dtype):
    return backend_dtype


def oracle(training, generated, holdout, schema, dtype, threshold):
    """Explicit training-fitted typed one-hot geometry and full tiny distance matrices."""
    (matrix, _) = distance_matrices(training, [generated, holdout], schema, dtype)
    cross = np.sqrt(matrix(1, 0))
    held_distances = np.sqrt(matrix(2, 0))
    forward = cross.min(axis=1)
    holdout_median = float(np.median(held_distances.min(axis=1)))
    values = {
        "median": float(np.median(forward)),
        "q05": float(np.quantile(forward, 0.05)),
        "ratio": float(np.median(forward)) / holdout_median if holdout_median else np.nan,
    }
    return (values, holdout_median)


@pytest.mark.parametrize("width", [1, 16])
def test_all_scores_match_independent_two_pool_oracle(backend, dtype, width):
    columns = [f"x{i}" for i in range(width)]
    r = pd.DataFrame(np.array([0.0, 2.0, 8.0, 16.0])[:, None] + np.arange(width), columns=columns)
    g = pd.DataFrame(np.array([1.0, 7.0, 20.0])[:, None] + np.arange(width), columns=columns)
    schema = Schema(numeric=columns)
    (original_r, original_g) = (r.copy(deep=True), g.copy(deep=True))
    h = r.iloc[[0, 1, 2, 3, 0]].copy() + 0.5
    outcome = compute(r, g, schema, backend, dtype, holdout=h)
    (expected, baseline) = oracle(r, g, h, schema, "float64" if backend == "python" else dtype, 0.5)
    assert isinstance(outcome, MemorizationOutcome)
    for key, value in expected.items():
        assert outcome.values[key] == pytest.approx(value, abs=1e-06, rel=2e-06)
    assert outcome.diagnostics["holdout_median"] == pytest.approx(baseline, rel=2e-06)
    assert outcome.diagnostics["usable_rows"] == {"training": 4, "generated": 3, "holdout": 5}
    assert not hasattr(outcome, "baselines")
    pd.testing.assert_frame_equal(r, original_r)
    pd.testing.assert_frame_equal(g, original_g)


@pytest.mark.parametrize("draws", [None, ([0, 0, 1], [0, 0, 0, 0])])
def test_positional_exclusion_retains_duplicate_neighbors(backend, dtype, draws):
    r = pd.DataFrame({"x": [0.0, 0.0, 8.0] if draws is None else [0.0, 8.0]})
    g = pd.DataFrame({"x": [0.0]})
    outcome = compute(r, g, Schema(numeric=["x"]), backend, dtype, draws=draws, holdout=r)
    assert outcome.values["median"] == 0.0
    assert outcome.diagnostics["holdout_median"] == 0.0
    assert outcome.diagnostics["zero_baseline"]
    assert np.isnan(outcome.values["ratio"])
    assert outcome.stat_flags == {"ratio": ["ZERO_BASELINE"]}


def test_single_training_row_keeps_raw_generated_statistics(backend, dtype):
    outcome = compute(
        pd.DataFrame({"x": [5.0]}),
        pd.DataFrame({"x": [5.0, 7.0]}),
        Schema(numeric=["x"]),
        backend,
        dtype,
    )
    assert outcome.values["median"] == 1.0
    assert outcome.values["q05"] == pytest.approx(0.1)
    assert np.isnan(outcome.values["ratio"])
    assert outcome.diagnostics["holdout_median"] is None
    assert outcome.stat_flags == {
        name: ["MISSING_HOLDOUT"]
        for name in ["ratio", *[f"holdout_q{p:02d}" for p in [5, 10, 25, 50, 75, 90, 95, 99]]]
    }


@pytest.mark.parametrize("side", ["training", "generated"])
@pytest.mark.parametrize("kind", ["empty", "missing", "numeric_text", "invalid_text", "infinity"])
def test_unavailable_sides_preserve_independent_training_baseline(backend, dtype, side, kind):
    pools = {
        "training": pd.DataFrame({"x": [0.0, 8.0]}),
        "generated": pd.DataFrame({"x": [0.0, 2.0]}),
    }
    values = {
        "empty": [],
        "missing": [np.nan],
        "numeric_text": ["2.0"],
        "invalid_text": ["bad"],
        "infinity": [np.inf],
    }[kind]
    pools[side] = pd.DataFrame({"x": values})
    if kind not in {"empty", "missing"}:
        with pytest.raises(ValidationError, match="declared numeric|numeric infinity"):
            compute(
                *pools.values(),
                Schema(numeric=["x"]),
                backend,
                dtype,
                holdout=pd.DataFrame({"x": [1.0, 3.0]}),
            )
        return
    outcome = compute(
        *pools.values(),
        Schema(numeric=["x"]),
        backend,
        dtype,
        holdout=pd.DataFrame({"x": [1.0, 3.0]}),
    )
    assert np.isnan(outcome.values["median"])
    if side == "training":
        assert outcome.diagnostics["holdout_median"] is None
    else:
        assert outcome.diagnostics["holdout_median"] == 0.25


def test_typed_categories_and_unknown_generated_levels(backend, dtype):
    r = pd.DataFrame({"c": pd.Series([1, "1", 1.0, "A"], dtype=object)})
    g = pd.DataFrame({"c": pd.Series([1, "unknown"], dtype=object)})
    schema = Schema(categorical=["c"])
    outcome = compute(r, g, schema, backend, dtype, holdout=g)
    (expected, baseline) = oracle(
        r, g, g, schema, "float64" if backend == "python" else dtype, 0.001
    )
    for key, value in expected.items():
        assert outcome.values[key] == pytest.approx(value, rel=2e-06)
    assert outcome.diagnostics["holdout_median"] == pytest.approx(baseline, rel=2e-06)


def test_complete_row_filtering_uses_only_declared_features(backend, dtype):
    outcome = compute(
        pd.DataFrame({"x": [0.0, 0.0, 8.0, np.nan], "unused": [1.0, 1.0, 1.0, 1.0]}),
        pd.DataFrame({"x": [0.0, 8.0], "unused": [1.0, np.nan]}),
        Schema(numeric=["x"]),
        backend,
        dtype,
    )
    assert outcome.diagnostics["usable_rows"] == {"training": 3, "generated": 2, "holdout": None}
    assert outcome.values["median"] == 0.0


def test_holdout_queries_preserve_nearest_distances_across_reference_blocks(backend, dtype):
    r = pd.DataFrame({"x": np.arange(131, dtype=float)})
    g = pd.DataFrame({"x": [0.0, 130.0]})
    outcome = compute(r, g, Schema(numeric=["x"]), backend, dtype, holdout=r + 0.25)
    assert outcome.diagnostics["holdout_median"] == pytest.approx(0.25 / 130, rel=1e-05)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_registered_self_tile_respects_global_offsets(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("CUDA unavailable")
    from tabgen_eval.execution.torch.distance.ops import nearest_squared_self_tile

    numeric = torch.tensor([[0.0], [0.0], [2.0], [4.0], [8.0]], dtype=torch.float64, device=device)
    categorical = torch.empty((5, 0), dtype=torch.int64, device=device)
    for qstart, qstop, rstart, rstop in [(0, 2, 0, 5), (2, 5, 0, 2), (1, 5, 2, 5)]:
        actual = nearest_squared_self_tile(
            numeric[qstart:qstop],
            numeric[rstart:rstop],
            categorical[qstart:qstop],
            categorical[rstart:rstop],
            qstart,
            rstart,
        )
        oracle = (
            (numeric[qstart:qstop, None, :] - numeric[None, rstart:rstop, :]).square().sum(dim=2)
        )
        for i in range(qstart, qstop):
            for j in range(rstart, rstop):
                if i == j:
                    oracle[i - qstart, j - rstart] = float("inf")
        torch.testing.assert_close(actual, oracle.amin(dim=1), rtol=0, atol=0)
    with pytest.raises(ValueError, match="offsets"):
        nearest_squared_self_tile(numeric, numeric, categorical, categorical, -1, 0)


def test_tensor_empty_batches_and_validation():
    pytest.importorskip("torch")
    r = pd.DataFrame({"x": [0.0, 8.0]})
    g = pd.DataFrame({"x": [0.0]})
    metric = DCR(device="cpu")
    context = context_for(metric, r, g, Schema(numeric=["x"]))
    empty = np.empty((0, 1), dtype=np.int64)
    assert context.compute_memorization(metric, empty, empty) == []
    with pytest.raises(ValueError, match="batch lengths"):
        context.compute_memorization(metric, np.array([[0], [1]]), np.array([[0]]))
    with pytest.raises(ValueError, match="index"):
        context.compute_memorization(metric, np.array([[2]]), np.array([[0]]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_batched_draws_fit_separate_training_geometries(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("CUDA unavailable")
    training = pd.DataFrame({"x": [0.0, 2.0, 8.0, 16.0]})
    generated = pd.DataFrame({"x": [1.0, 15.0]})
    metric = DCR(device=device, dtype="float64")
    context = context_for(metric, training, generated, Schema(numeric=["x"]), n_jobs=4)
    outcomes = context.compute_memorization(
        metric, np.array([[0, 1, 2], [1, 2, 3]]), np.array([[0], [1]])
    )
    assert len(outcomes) == 2
    assert all(outcome.diagnostics["holdout_median"] is None for outcome in outcomes)
    assert outcomes[0].values["median"] == 0.125
    assert outcomes[1].values["median"] == pytest.approx(1 / 14)
    assert all(np.isnan(outcome.values["ratio"]) for outcome in outcomes)


def test_registered_self_tile_fake_contract():
    torch = pytest.importorskip("torch")
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tabgen_eval.execution.torch.distance.ops import nearest_squared_self_tile

    with FakeTensorMode():
        numeric = torch.empty((7, 3), dtype=torch.float64)
        categorical = torch.empty((7, 2), dtype=torch.int64)
        output = nearest_squared_self_tile(
            numeric[:3], numeric[2:], categorical[:3], categorical[2:], 0, 2
        )
        assert output.shape == (3,)
        assert output.dtype == torch.float64
        assert output.device.type == "cpu"


@pytest.mark.parametrize("kind", ["missing", "empty", "filtered", "zero", "positive"])
def test_holdout_availability_and_single_training_row(backend, dtype, kind):
    from tests.support.evaluation import memorization_report as evaluate_memorization

    training = pd.DataFrame({"x": [5.0]})
    generated = pd.DataFrame({"x": [5.0, 7.0]})
    held = {
        "missing": None,
        "empty": pd.DataFrame({"x": pd.Series([], dtype=float)}),
        "filtered": pd.DataFrame({"x": [np.nan]}),
        "zero": pd.DataFrame({"x": [5.0]}),
        "positive": pd.DataFrame({"x": [6.0, 8.0]}),
    }[kind]
    report = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=held,
        options=make_options(
            device="cpu" if backend == "python" else backend,
            dtype=dtype,
        ),
    )
    outcome = report.results[0]
    assert report["dcr/median"].value == 1.0
    assert report["dcr/q05"].value == pytest.approx(0.1)
    if kind == "positive":
        assert report["dcr/ratio"].value == 0.5
        assert outcome.diagnostics["holdout_median"] == 2.0
    else:
        expected = (
            "MISSING_HOLDOUT"
            if kind == "missing"
            else "ZERO_BASELINE"
            if kind == "zero"
            else "INSUFFICIENT_HOLDOUT_ROWS"
        )
        assert np.isnan(report["dcr/ratio"].value)
        assert expected in report["dcr/ratio"].flag_codes
        assert outcome.diagnostics["holdout_median"] == (0.0 if kind == "zero" else None)


def test_holdout_label_types_and_outliers_cannot_change_training_generated_scores(backend, dtype):
    from tests.support.evaluation import memorization_report as evaluate_memorization

    training = pd.DataFrame({"x": [5.0, 5.0], "c": [1.0, 2.0]})
    generated = pd.DataFrame({"x": [5.0, 6.0, 5.0], "c": [1, 1, 2]})
    schema = Schema(numeric=["x"], categorical=["c"])
    options = make_options(
        device="cpu" if backend == "python" else backend,
        dtype=dtype,
    )
    scores = []
    for held in [
        None,
        pd.DataFrame({"x": [5.0], "c": [1.0]}),
        pd.DataFrame({"x": [10000000000.0, np.nan], "c": ["1", "new"]}),
    ]:
        report = evaluate_memorization(training, generated, schema, holdout=held, options=options)
        scores.append([report[key].value for key in ["dcr/median", "dcr/q05"]])
    np.testing.assert_allclose(scores, np.repeat([scores[0]], 3, axis=0), rtol=0, atol=0)
    assert scores[0][0] == 1.0
