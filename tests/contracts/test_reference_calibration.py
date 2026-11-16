"""Independent sampling and numerical contracts for real-only calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.sampling import SamplingPlan, draw_batches
from tabgen_eval.metrics import PRDC, Conditional, LinearC2ST, Shape, Trend
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def _calibrate_metrics(real, schema=None, *, metrics, options=None, **kwargs):
    """Run numerical fixtures directly through the private reference engine."""
    options = options or make_options()
    return calibrate(
        real,
        schema,
        options=options,
        metrics=metrics,
        **kwargs,
    )


def _draws(n_source, n_replicates, seed):
    """Independent ordered-pair oracle using the declared row RNG contract."""
    rng = np.random.default_rng(seed)
    n = n_source // 2
    pairs = [rng.choice(n_source, size=2 * n, replace=False) for _ in range(n_replicates)]
    return np.asarray([p[:n] for p in pairs]), np.asarray([p[n:] for p in pairs])


def _collect(plan, n_replicates, seed, batch_size):
    batches = list(draw_batches(plan, n_replicates=n_replicates, seed=seed, batch_size=batch_size))
    return np.concatenate([b.reference for b in batches]), np.concatenate(
        [b.candidate for b in batches]
    )


@pytest.mark.parametrize("n_source", [23, 41, 3])
def test_ordered_draws_match_independent_full_pool_oracle(n_source):
    plan = SamplingPlan.create(
        n_source,
    )
    assert plan.n_source == n_source
    assert plan.n_res == n_source // 2
    expected = _draws(n_source, 13, 731)
    for batch_size in [1, 4, 20]:
        actual = _collect(plan, 13, 731, batch_size)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)
        for reference, candidate in zip(*actual, strict=True):
            assert len(set(reference) | set(candidate)) == 2 * plan.n_res
    short = _collect(plan, 5, 731, 3)
    for prefix, full in zip(short, expected, strict=True):
        np.testing.assert_array_equal(prefix, full[:5])


def test_automatic_resolution_keeps_all_source_rows_eligible():
    plan = SamplingPlan.create(
        101,
    )
    reference, candidate = _collect(plan, 20, 14, 7)
    assert reference.shape == candidate.shape == (20, 50)
    # A fixed ten-row source subset could never expose more than ten identities.
    assert len(np.unique(np.concatenate([reference.ravel(), candidate.ravel()]))) > 10
    assert np.max([reference.max(), candidate.max()]) >= 10


@pytest.mark.parametrize("budget", [True, False, 0, 1, -2, 24, 2.5, "10"])
def test_removed_budget_fails_before_sampling(budget):
    with pytest.raises((TypeError, ValueError)):
        SamplingPlan.create(23, sample_budget=budget)


def _frame(n=241):
    rng = np.random.default_rng(731)
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": 0.6 * x + rng.normal(scale=0.4, size=n),
            "z": rng.uniform(-2, 2, n),
            "c": rng.choice(["a", "b", "nan", " A "], n),
        },
        index=np.repeat(7, n),
    )
    return frame, Schema(numeric=["x", "y", "z"], categorical=["c"])


@pytest.mark.parametrize(
    "metric",
    [
        Shape(),
        Trend(),
        Conditional(min_count=2, num_bins=4, rare_threshold=0.2),
        PRDC(nearest_k=3),
    ],
    ids=lambda metric: metric.name,
)
def test_raw_replicates_match_manual_indexed_metric_evaluation(metric):
    frame, schema = _frame()
    options = make_options(seed=19)
    result = _calibrate_metrics(
        frame,
        schema,
        metrics=[metric],
        n_replicates=5,
        seed=103,
        options=options,
    )
    reference, candidate = _draws(len(frame), 5, 103)
    expected = [
        metric.compute(frame.iloc[a], frame.iloc[b], schema, rng=np.random.default_rng(0))
        for a, b in zip(reference, candidate, strict=True)
    ]
    observed = result.metrics[metric.name]
    for name in metric.stat_names:
        values = np.asarray([entry[name] for entry in expected])
        statistic = summaries(observed)[name]
        np.testing.assert_allclose(statistic.samples, values, equal_nan=True, atol=1e-12)
        finite = values[np.isfinite(values)]
        assert statistic.finite_count == len(finite)
        assert statistic.unavailable_count == len(values) - len(finite)
        assert not statistic.samples.flags.writeable
        if len(finite):
            np.testing.assert_allclose(
                [statistic.mean, statistic.median], [np.mean(finite), np.median(finite)]
            )
            np.testing.assert_allclose(
                [statistic.band_low, statistic.band_high],
                np.quantile(finite, [0.025, 0.975]),
            )
    assert result.metadata["n_res"] == 120
    assert "sample_budget" not in result.metadata
    assert result.metadata["n_source"] == 241
    assert result.metadata["seed"] == 103
    assert result.metadata["confidence_interval"] == 0.95


def test_odd_source_and_tiny_budget_are_computed_without_an_advisory_size_gate():
    frame = pd.DataFrame({"x": np.arange(23, dtype=float)})
    result = _calibrate_metrics(
        frame,
        Schema(numeric=["x"]),
        metrics=[Shape()],
        n_replicates=4,
        seed=7,
        options=make_options(),
    )
    assert "sample_budget" not in result.metadata
    assert result.metadata["n_res"] == 11
    assert summaries(result.metrics["shape"])["shape"].finite_count == 4
    tiny = _calibrate_metrics(
        frame.iloc[:3],
        Schema(numeric=["x"]),
        metrics=[Shape()],
        n_replicates=2,
        seed=7,
        options=make_options(),
    )
    assert tiny.metadata["n_res"] == 1
    assert summaries(tiny.metrics["shape"])["shape"].finite_count == 2


def test_metric_order_and_additional_metrics_do_not_change_row_or_fit_streams():
    frame, schema = _frame()
    common = dict(n_replicates=3, seed=37, options=make_options())
    first = _calibrate_metrics(
        frame,
        schema,
        metrics=[Shape(), Trend()],
        **common,
    )
    reversed_order = _calibrate_metrics(
        frame,
        schema,
        metrics=[Trend(), Shape()],
        **common,
    )
    alone = _calibrate_metrics(frame, schema, metrics=[Trend()], **common)
    for result in [first, reversed_order]:
        for stat in Trend().stat_names:
            np.testing.assert_array_equal(
                summaries(result.metrics["trend"])[stat].samples,
                summaries(alone.metrics["trend"])[stat].samples,
            )


def test_conditional_applicability_is_decided_on_each_reference_draw():
    n = 240
    frame = pd.DataFrame(
        {
            "c1": ["minor"] * 48 + ["major"] * 192,
            "c2": np.tile(["a", "b"], n // 2),
            "c3": np.tile(["a", "a", "b", "b"], n // 4),
        }
    )
    schema = Schema(categorical=list(frame.columns))
    metric = Conditional(min_count=2, rare_threshold=0.2)
    assert {"cond_shape_rare", "cond_trend_rare", "recovery_rare"} <= metric.not_applicable(
        frame, frame, schema
    )
    result = _calibrate_metrics(
        frame,
        schema,
        metrics=[metric],
        n_replicates=12,
        seed=17,
        options=make_options(),
    )
    statistic = summaries(result.metrics[metric.name])["recovery_rare"]
    assert statistic.finite_count > 0
    assert statistic.unavailable_count > 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_matches_python_with_exact_same_draws(dtype):
    pytest.importorskip("torch")
    frame, schema = _frame()
    common = dict(n_replicates=3, seed=23, options=make_options())
    python = _calibrate_metrics(frame, schema, metrics=[Shape()], **common)
    cpu = _calibrate_metrics(
        frame,
        schema,
        metrics=[Shape(device="cpu", dtype=dtype)],
        **common,
    )
    np.testing.assert_allclose(
        summaries(cpu.metrics["shape"])["shape"].samples,
        summaries(python.metrics["shape"])["shape"].samples,
        atol=1e-6 if dtype == "float32" else 1e-12,
    )


def test_classifier_does_not_fit_preprocessing_on_undrawn_source_rows():
    pytest.importorskip("torch")
    frame, schema = _frame(401)
    reference, candidate = _draws(len(frame), 1, 17)
    drawn = np.union1d(reference[0], candidate[0])
    undrawn = np.setdiff1d(np.arange(len(frame)), drawn)
    changed = frame.copy()
    changed.iloc[undrawn, :3] = 1e12
    changed.iloc[undrawn, 3] = "only-outside-the-comparison"
    common = dict(n_replicates=1, seed=17, options=make_options())
    options = dict(device="cpu", solver="direct", tolerance=1e-06)
    original = _calibrate_metrics(frame, schema, metrics=[LinearC2ST(**options)], **common)
    perturbed = _calibrate_metrics(changed, schema, metrics=[LinearC2ST(**options)], **common)
    first = summaries(original.metrics["linear_c2st"])["auc"]
    second = summaries(perturbed.metrics["linear_c2st"])["auc"]
    assert first.finite_count == second.finite_count == 1
    np.testing.assert_array_equal(first.samples, second.samples)


def test_python_classifier_receives_draw_local_folds_and_point_fit_policy(monkeypatch):
    from tabgen_eval.execution.classifier.protocol import source_folds
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome

    frame, schema = _frame()
    frame["source_position"] = np.arange(len(frame))
    reference, candidate = _draws(len(frame), 4, 137)
    calls = []

    def record(
        self,
        real,
        other,
        declared,
        reference_folds,
        candidate_folds,
        *,
        seed,
        point=False,
        fit_threads=1,
    ):
        import hashlib

        replicate = len(calls)
        key = int.from_bytes(hashlib.sha256(self.name.encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence(137, spawn_key=(key, replicate, 0xC2A1)))
        expected_folds = source_folds(120, 120, self.cv_folds, rng)
        expected_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        assert point is True
        assert seed == expected_seed
        assert declared == schema
        np.testing.assert_array_equal(real.source_position, reference[replicate])
        np.testing.assert_array_equal(other.source_position, candidate[replicate])
        np.testing.assert_array_equal(reference_folds, expected_folds[0])
        np.testing.assert_array_equal(candidate_folds, expected_folds[1])
        assert set(real.source_position).isdisjoint(other.source_position)
        calls.append(seed)
        return ClassifierOutcome(0.5, [{"converged": True}])

    monkeypatch.setattr(LinearC2ST, "compute_partitioned", record)
    result = _calibrate_metrics(
        frame,
        schema,
        metrics=[LinearC2ST()],
        n_replicates=4,
        seed=137,
        options=make_options(n_jobs=1),
    )
    assert len(calls) == 4
    np.testing.assert_array_equal(
        summaries(result.metrics["linear_c2st"])["auc"].samples, np.full(4, 0.5)
    )


def test_native_classifier_uses_point_policy_for_each_fresh_comparison(monkeypatch):
    pytest.importorskip("torch")
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
    from tabgen_eval.execution.torch.context import TorchContext

    frame, schema = _frame()
    calls = []

    def record(
        self,
        metric,
        reference,
        candidate,
        rf,
        sf,
        seeds,
        *,
        ideal=False,
        point=False,
        fit_threads=None,
    ):
        assert ideal is True
        assert point is True
        assert fit_threads == 1
        for a, b, f, g in zip(reference, candidate, rf, sf, strict=True):
            assert len(set(a) | set(b)) == len(a) + len(b)
            assert len(f) == len(a)
            assert len(g) == len(b)
        calls.extend(seeds)
        return [ClassifierOutcome(0.5, [{"converged": True}]) for _ in seeds]

    monkeypatch.setattr(TorchContext, "compute_classifier", record)
    result = _calibrate_metrics(
        frame,
        schema,
        metrics=[LinearC2ST(device="cpu")],
        n_replicates=3,
        seed=137,
        options=make_options(),
    )
    assert len(calls) == 3
    assert summaries(result.metrics["linear_c2st"])["auc"].finite_count == 3


def test_finite_nonconverged_classifier_scores_are_retained_with_diagnostics(monkeypatch):
    pytest.importorskip("torch")
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
    from tabgen_eval.execution.torch.context import TorchContext

    frame, schema = _frame()
    seen = 0

    def outcomes(self, metric, reference, candidate, rf, sf, seeds, **kwargs):
        nonlocal seen
        result = []
        for _ in seeds:
            converged = seen != 1
            result.append(
                ClassifierOutcome(0.75 if not converged else 0.5, [{"converged": converged}])
            )
            seen += 1
        return result

    monkeypatch.setattr(TorchContext, "compute_classifier", outcomes)
    result = _calibrate_metrics(
        frame,
        schema,
        metrics=[LinearC2ST(device="cpu")],
        n_replicates=3,
        seed=137,
        options=make_options(),
    )
    statistic = summaries(result.metrics["linear_c2st"])["auc"]
    assert statistic.finite_count == 3
    assert statistic.unavailable_count == 0
    np.testing.assert_array_equal(statistic.samples, [0.5, 0.75, 0.5])
    assert statistic.mean == pytest.approx(1.75 / 3)
    assert statistic.median == statistic.band_low == 0.5
    assert statistic.band_high == pytest.approx(0.7375)


def test_config_seed_and_default_replicate_count_are_recorded_and_replayable():
    frame = pd.DataFrame({"x": np.arange(23, dtype=float)})
    result = _calibrate_metrics(
        frame,
        Schema(numeric=["x"]),
        metrics=[Shape()],
        options=make_options(seed=987),
    )
    assert result.metadata["seed"] == 987
    assert result.metadata["n_replicates"] == 10
    reference, candidate = _draws(len(frame), 10, 987)
    metric = Shape()
    expected = [
        metric.compute(frame.iloc[a], frame.iloc[b], Schema(numeric=["x"]), rng=None)["shape"]
        for a, b in zip(reference, candidate, strict=True)
    ]
    np.testing.assert_array_equal(summaries(result.metrics["shape"])["shape"].samples, expected)


def test_native_classifier_scores_match_manual_partitioned_point_evaluations():
    pytest.importorskip("torch")
    import hashlib

    from tabgen_eval.execution.classifier.protocol import source_folds
    from tabgen_eval.execution.runtime.dispatch import context_for

    frame, schema = _frame()
    metric = LinearC2ST(device="cpu", solver="direct")
    seed = 73
    result = _calibrate_metrics(
        frame,
        schema,
        metrics=[metric],
        n_replicates=3,
        seed=seed,
        options=make_options(),
    )
    reference, candidate = _draws(len(frame), 3, seed)
    metric_id = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    rf, sf, seeds = [], [], []
    for replicate in range(3):
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(metric_id, replicate, 0xC2A1))
        )
        a, b = source_folds(120, 120, metric.cv_folds, rng)
        rf.append(a)
        sf.append(b)
        seeds.append(int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)))
    context = context_for(metric, frame, frame, schema)
    outcomes = context.compute_classifier(
        metric,
        reference,
        candidate,
        np.array(rf),
        np.array(sf),
        seeds,
        ideal=True,
        point=True,
    )
    assert all(outcome.converged for outcome in outcomes)
    np.testing.assert_array_equal(
        summaries(result.metrics[metric.name])["auc"].samples,
        [outcome.value for outcome in outcomes],
    )


def test_classifier_order_and_worker_budget_preserve_fixed_draw_and_fit_streams():
    pytest.importorskip("torch")
    frame, schema = _frame()
    first = _calibrate_metrics(
        frame,
        schema,
        metrics=[LinearC2ST(device="cpu", solver="direct")],
        n_replicates=3,
        seed=97,
        options=make_options(n_jobs=1),
    )
    reordered = _calibrate_metrics(
        frame,
        schema,
        metrics=[
            Shape(),
            LinearC2ST(device="cpu", solver="direct"),
        ],
        n_replicates=3,
        seed=97,
        options=make_options(n_jobs=2),
    )
    np.testing.assert_array_equal(
        summaries(first.metrics["linear_c2st"])["auc"].samples,
        summaries(reordered.metrics["linear_c2st"])["auc"].samples,
    )
