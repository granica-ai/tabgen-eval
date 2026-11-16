"""Independent oracles for complete-pool paired resampling and score execution."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval._evaluation.estimator import Estimator
from tabgen_eval.execution.runtime.sampling import fixed_draw, paired_draw_batches
from tabgen_eval.metrics import Conditional
from tabgen_eval.metrics.base import Direction, Metric, StatSpec
from tests.support.options import make_options


class RecordingMetric(Metric):
    name = "sampling_probe"
    category = "test"
    backend = "torch"
    stats = (StatSpec("score", "Score", Direction.HIGHER_BETTER),)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def compute(self, reference, candidate, schema, *, rng):
        self.calls.append((reference.x.to_numpy().copy(), candidate.x.to_numpy().copy()))
        return {"score": float(reference.x.mean() + candidate.x.mean() + rng.uniform())}


def batches(draw, count=7, size=3, generated=True):
    return list(
        paired_draw_batches(
            draw, n_replicates=count, seed=17, batch_size=size, include_generated=generated
        )
    )


def flattened(draw, count=7, size=3, generated=True):
    result = batches(draw, count, size, generated)
    return tuple(
        np.concatenate([getattr(b, key) for b in result])
        for key in ("reference", "real_candidate", "generated")
        if generated or key != "generated"
    )


@pytest.mark.parametrize("n_generated,n_res", [(23, 18), (11, 5)])
def test_complete_pool_draw_oracle_and_independent_generated_stream(n_generated, n_res):
    draw = fixed_draw(37, n_generated, seed=17, uncertainty="both")
    r, c, g = flattened(draw)
    real_rng = np.random.default_rng(np.random.SeedSequence(17, spawn_key=(0xB007, 0)))
    fake_rng = np.random.default_rng(np.random.SeedSequence(17, spawn_key=(0xB007, 1)))
    for i in range(7):
        expected = real_rng.choice(37, 2 * n_res, replace=False)
        np.testing.assert_equal(r[i], expected[:n_res])
        np.testing.assert_equal(c[i], expected[n_res:])
        np.testing.assert_equal(g[i], fake_rng.choice(n_generated, n_res, replace=False))
        assert len(set(r[i]) | set(c[i])) == 2 * n_res
        assert len(np.unique(g[i])) == n_res
    assert np.any(r >= 17) and np.any(c >= 17)
    assert not np.array_equal(r[0], r[1])
    assert np.any(g >= n_res)
    assert len({tuple(sorted(rows)) for rows in g}) > 1


@pytest.mark.parametrize(
    "n_holdout,n_generated,expected",
    [
        (1000, 400, (400, 200)),
        (1000, 500, (500, 250)),
        (1000, 502, (500, 500)),
        (1000, 700, (500, 500)),
        (11, 5, (5, 2)),
        (11, 6, (5, 5)),
        (10, 1, (1, 0)),
    ],
)
def test_mode_specific_resolution_boundaries(n_holdout, n_generated, expected):
    fixed, both = expected
    for mode, n_res in (("none", fixed), ("reference", fixed), ("candidate", both), ("both", both)):
        draw = fixed_draw(n_holdout, n_generated, seed=17, uncertainty=mode)
        assert draw.n_res == n_res
        assert draw.n_holdout == n_holdout and draw.n_generated == n_generated
        assert len(set(draw.reference) | set(draw.real_candidate)) == 2 * n_res
        assert len(set(draw.generated)) == n_res
        assert all(
            not rows.flags.writeable
            for rows in (draw.reference, draw.real_candidate, draw.generated)
        )


@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("n_generated", [11, 23])
def test_prefix_batch_mode_and_global_rng_invariance(size, n_generated):
    draw = fixed_draw(37, n_generated, seed=17, uncertainty="both")
    np.random.seed(99)
    before = np.random.get_state()
    expected = flattened(draw)
    actual = flattened(draw, count=10, size=size)
    real_only = flattened(draw, count=10, size=size, generated=False)
    after = np.random.get_state()
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_equal(a, b[:7])
    for a, b in zip(actual[:2], real_only, strict=True):
        np.testing.assert_equal(a, b)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_equal(a, b)


def test_all_generated_rows_keep_order_with_odd_holdout():
    draw = fixed_draw(
        31,
        7,
    )
    assert draw.n_res == 7 and draw.n_holdout == 31
    for g in flattened(draw)[2]:
        np.testing.assert_equal(g, np.arange(7))


@pytest.mark.parametrize(
    "mode,count", [("none", 2), ("reference", 5), ("candidate", 5), ("both", 8)]
)
def test_comparison_counts_shared_references_and_repetition_means(mode, count):
    h = pd.DataFrame({"x": np.arange(37)})
    g = pd.DataFrame({"x": 1000 + np.arange(23)})
    metric = RecordingMetric()
    cfg = make_options(uncertainty=mode, n_replicates=4, n_jobs=1)
    result = Estimator(metric, cfg).run(h, g, Schema(numeric=["x"]))["score"]
    assert len(metric.calls) == count
    generated_calls = [(r, c) for r, c in metric.calls if np.all(c >= 1000)]
    reference_calls = [(r, c) for r, c in metric.calls if np.all(c < 1000)]
    assert all(not np.intersect1d(r, c).size for r, c in reference_calls)
    if mode == "none":
        np.testing.assert_array_equal(generated_calls[0][0], reference_calls[0][0])
        assert result.generated_distribution is result.reference_distribution is None
    else:
        assert len(reference_calls) == (1 if mode == "candidate" else 4)
        if mode != "candidate":
            assert result.ideal == result.reference_distribution.mean
        else:
            assert result.reference_distribution is None
            assert result.ideal_estimate == "point"
        assert len(generated_calls) == (1 if mode == "reference" else 4)
        if mode != "reference":
            assert result.value == result.generated_distribution.mean
            if mode == "both":
                for (r, _), (rr, _) in zip(generated_calls, reference_calls, strict=True):
                    np.testing.assert_array_equal(r, rr)
        else:
            fixed_metric = RecordingMetric()
            fixed = Estimator(fixed_metric, make_options()).run(h, g, Schema(numeric=["x"]))[
                "score"
            ]
            assert result.value == fixed.value


def test_python_worker_count_preserves_scores_and_replicate_prefix():
    h = pd.DataFrame({"x": np.arange(37)})
    g = pd.DataFrame({"x": 1000 + np.arange(23)})
    outcomes = []
    for jobs, reps in [(1, 4), (2, 7)]:
        outcomes.append(
            Estimator(
                RecordingMetric(),
                make_options(
                    n_jobs=jobs,
                    uncertainty="both",
                    n_replicates=reps,
                ),
            ).run(h, g, Schema(numeric=["x"]))["score"]
        )
    assert all(outcome.value == outcome.generated_distribution.mean for outcome in outcomes)
    np.testing.assert_equal(
        outcomes[0].generated_distribution.samples, outcomes[1].generated_distribution.samples[:4]
    )
    np.testing.assert_equal(
        outcomes[0].reference_distribution.samples, outcomes[1].reference_distribution.samples[:4]
    )


@pytest.mark.parametrize(
    "mode,n_res", [("none", 7), ("reference", 7), ("candidate", 3), ("both", 3)]
)
def test_point_and_repeated_comparisons_use_the_same_mode_resolution(mode, n_res):
    metric = RecordingMetric()
    result = Estimator(
        metric,
        make_options(uncertainty=mode, n_replicates=4, seed=17, n_jobs=1),
    ).run(
        pd.DataFrame({"x": np.arange(31)}),
        pd.DataFrame({"x": 1000 + np.arange(7)}),
        Schema(numeric=["x"]),
    )["score"]
    assert result.n_eff == n_res
    assert all(len(reference) == len(candidate) == n_res for reference, candidate in metric.calls)
    generated_calls = [candidate for _, candidate in metric.calls if np.all(candidate >= 1000)]
    if mode in {"candidate", "both"}:
        assert len(generated_calls) == 4
        assert len({tuple(sorted(rows)) for rows in generated_calls}) > 1
        assert any(np.any(rows >= 1000 + n_res) for rows in generated_calls)
    else:
        assert len(generated_calls) == 1
        np.testing.assert_array_equal(generated_calls[0], 1000 + np.arange(7))


@pytest.mark.parametrize("mode", ["candidate", "both"])
def test_single_generated_row_with_candidate_uncertainty_is_insufficient(mode):
    metric = RecordingMetric()
    result = Estimator(metric, make_options(uncertainty=mode, n_replicates=3)).run(
        pd.DataFrame({"x": np.arange(10)}),
        pd.DataFrame({"x": [1000]}),
        Schema(numeric=["x"]),
    )["score"]
    assert not metric.calls
    assert result.n_eff == 0
    assert np.isnan(result.value) and np.isnan(result.ideal)
    assert "INSUFFICIENT_DATA" in result.flag_codes
    assert result.generated_distribution.unavailable_count == 3
    if mode == "both":
        assert result.reference_distribution.unavailable_count == 3
    else:
        assert result.reference_distribution is None


@pytest.mark.parametrize("name", ["shape", "trend", "conditional_fidelity", "prdc"])
def test_cpu_batched_replicates_match_independent_draw_numerics(name):
    pytest.importorskip("torch")
    from tabgen_eval.registry import resolve_metrics
    from tests.support.evaluation import paired_report as evaluate

    rng = np.random.default_rng(40)
    holdout = pd.DataFrame(
        {
            "x": rng.normal(size=123),
            "y": rng.normal(size=123),
            "c": rng.choice(["a", "b", "rare"], 123),
        }
    )
    generated = pd.DataFrame(
        {"x": rng.normal(0.5, size=81), "y": rng.normal(size=81), "c": rng.choice(["a", "b"], 81)}
    )
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    options = make_options(
        device="cpu",
        uncertainty="both",
        n_replicates=3,
        seed=17,
    )
    report = evaluate(holdout, generated, schema, metrics=[name], options=options)
    metric = resolve_metrics([name], options)[0]
    draw = fixed_draw(123, 81, seed=17)
    reference, candidate, fake = flattened(draw, count=3)
    for i in range(3):
        for ideal, rows in [(False, fake), (True, candidate)]:
            source = holdout if ideal else generated
            expected = metric.compute(
                holdout.iloc[reference[i]].reset_index(drop=True),
                source.iloc[rows[i]].reset_index(drop=True),
                schema,
                rng=np.random.default_rng(0),
            )
            for stat_name, value in expected.items():
                actual = report[f"{name}/{stat_name}"]
                scores = (
                    actual.reference_distribution.samples
                    if ideal
                    else actual.generated_distribution.samples
                )
                np.testing.assert_allclose(scores[i], value, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_explicit_custom_minimum_skips_only_noncomputable_draws():
    class RequiredRows(RecordingMetric):
        n_min = 12

    metric = RequiredRows()
    result = Estimator(metric, make_options(uncertainty="both", n_replicates=3)).run(
        pd.DataFrame({"x": np.arange(20)}),
        pd.DataFrame({"x": np.arange(20)}),
        Schema(numeric=["x"]),
    )["score"]
    assert not metric.calls
    assert np.isnan(result.value) and np.isnan(result.ideal)
    assert "INSUFFICIENT_DATA" in result.flag_codes
    assert result.generated_distribution.unavailable_count == 3
    assert result.reference_distribution.unavailable_count == 3


def test_builtin_computable_small_sample_is_not_suppressed():
    from tests.support.evaluation import paired_report as evaluate

    real = pd.DataFrame({"x": np.arange(12.0)})
    result = evaluate(
        real,
        real.copy(),
        Schema(numeric=["x"]),
        metrics=["shape"],
        options=make_options(uncertainty="both", n_replicates=3),
    )["shape/shape"]
    assert result.n_eff == 6
    assert np.isfinite(result.value) and np.isfinite(result.ideal)
    assert np.isfinite(result.ci_low) and np.isfinite(result.ideal_ci_low)


def test_shared_draw_can_be_supplied_without_resampling(monkeypatch):
    import tabgen_eval._evaluation.estimator as engine

    draw = fixed_draw(37, 23, seed=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("shared source selection must not repeat per metric")

    monkeypatch.setattr(engine, "eligible_fixed_draw", forbidden)
    metric = RecordingMetric()
    Estimator(metric, make_options(seed=4), draw=draw).run(
        pd.DataFrame({"x": np.arange(37)}),
        pd.DataFrame({"x": np.arange(23) + 1000}),
        Schema(numeric=["x"]),
    )
    np.testing.assert_equal(metric.calls[0][0], draw.reference)


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("details", [False, True])
def test_repeated_rare_slice_applicability_follows_finite_draws(varied, details):
    pytest.importorskip("torch")
    holdout = pd.DataFrame(
        {"a": ["a"] * 20 + ["b"] * 20, "b": ["x", "y"] * 20}
        if varied
        else {"a": ["a"] * 40, "b": ["x"] * 40}
    )
    generated = holdout.iloc[::2].reset_index(drop=True)
    schema = Schema(categorical=["a", "b"])
    metric = Conditional(rare_threshold=0.4, min_count=1, num_bins=2, device="cpu")
    # Neither feature has rare categories in the full holdout. Sampling can
    # nevertheless create eligible rare slices in individual repetitions.
    assert "cond_shape_rare" in metric.not_applicable(holdout, generated, schema)
    result = Estimator(
        metric,
        make_options(uncertainty="both", n_replicates=20, n_jobs=1, device="cpu"),
        details=details,
    ).run(holdout, generated, schema)
    for name in ("cond_shape_rare", "recovery_rare"):
        stat = result[name]
        distribution = stat.generated_distribution
        assert distribution.requested_count == 20
        if varied:
            assert 0 < distribution.finite_count < distribution.requested_count
            assert np.isfinite(stat.value)
            assert stat.value == distribution.mean
            assert "NOT_APPLICABLE" not in stat.flag_codes
            assert "NONFINITE_REPLICATES" in stat.flag_codes
        else:
            assert distribution.finite_count == 0
            assert np.isnan(stat.value)
            assert "NOT_APPLICABLE" in stat.flag_codes
    # Two columns never supply the three needed by conditional Trend.
    assert "NOT_APPLICABLE" in result["cond_trend_all"].flag_codes
