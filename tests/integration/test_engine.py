"""Matched points, paired score distributions, and factual diagnostics."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import DiagnosticFlag
from tabgen_eval._evaluation.estimator import Estimator
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tabgen_eval.metrics.base import Direction, Metric, StatSpec
from tabgen_eval.schema import Schema
from tests.support.options import make_options

pytestmark = pytest.mark.unit
SCHEMA = Schema(numeric=["x"])


class MeanDiff(Metric):
    """Bounded mean similarity with an explicit two-row requirement."""

    name = "meandiff"
    category = "fidelity"
    n_min = 2
    stats = (StatSpec("score", "Score", Direction.HIGHER_BETTER),)

    def compute(self, reference, candidate, schema, *, rng):
        return {"score": 1 - min(abs(reference.x.mean() - candidate.x.mean()), 1)}


def _frame(n, loc, seed):
    return pd.DataFrame({"x": np.random.default_rng(seed).normal(loc, 0.1, n)})


@pytest.mark.parametrize(
    "nr,ng,n",
    [
        (100, 60, 50),
        (100, 30, 30),
        (101, 60, 50),
        (100, 60, 50),
        (101, 60, 50),
    ],
)
def test_pool_sizes_define_resolution(nr, ng, n):
    draw = fixed_draw(nr, ng, seed=3)
    assert draw.n_res == n
    assert len(set(draw.reference) | set(draw.real_candidate)) == 2 * n
    assert len(set(draw.generated)) == n


def test_point_comparisons_share_reference_and_use_disjoint_real_roles():
    class Recording(MeanDiff):
        def __init__(self):
            super().__init__()
            self.calls = []

        def compute(self, reference, candidate, schema, *, rng):
            self.calls.append((reference.x.to_numpy(), candidate.x.to_numpy()))
            return super().compute(reference, candidate, schema, rng=rng)

    metric = Recording()
    real = pd.DataFrame({"x": np.arange(1000.0)})
    generated = pd.DataFrame({"x": np.arange(5000.0, 5800.0)})
    Estimator(
        metric,
        make_options(
            seed=1,
        ),
    ).run(real, generated, SCHEMA)
    (r, g), (same_r, other_r) = metric.calls
    np.testing.assert_array_equal(r, same_r)
    assert len(r) == len(g) == len(other_r) == 500
    assert set(r).isdisjoint(other_r)
    assert set(r) <= set(real.x) and set(other_r) <= set(real.x)
    assert set(g) <= set(generated.x)


@pytest.mark.parametrize("mode", ["reference", "both"])
def test_reported_bands_are_unshifted_empirical_quantiles(mode):
    real, generated = _frame(300, 0, 1), _frame(300, 0.2, 2)
    cfg = make_options(uncertainty=mode, n_replicates=20, seed=5)
    stat = Estimator(MeanDiff(), cfg).run(real, generated, SCHEMA)["score"]
    reference = stat.reference_distribution
    np.testing.assert_array_equal(
        [stat.ideal_ci_low, stat.ideal_ci_high], np.quantile(reference.samples, [0.025, 0.975])
    )
    assert stat.ideal_bootstrap_median == np.median(reference.samples)
    if mode == "both":
        np.testing.assert_array_equal(
            [stat.ci_low, stat.ci_high],
            np.quantile(stat.generated_distribution.samples, [0.025, 0.975]),
        )
    else:
        assert stat.generated_distribution is None
        assert np.isnan(stat.ci_low) and np.isnan(stat.ci_high)


def test_means_and_reference_prefix_follow_the_requested_repetitions():
    real, generated = _frame(300, 0, 1), _frame(300, 0.2, 2)
    results = [
        Estimator(MeanDiff(), make_options(uncertainty=mode, n_replicates=b, seed=5)).run(
            real, generated, SCHEMA
        )["score"]
        for mode, b in [("none", 10), ("reference", 10), ("both", 10), ("both", 20)]
    ]
    assert results[0].value == results[1].value
    for s in results[1:]:
        assert s.ideal == s.reference_distribution.mean
    for s in results[2:]:
        assert s.value == s.generated_distribution.mean
    np.testing.assert_array_equal(
        results[1].reference_distribution.samples, results[2].reference_distribution.samples
    )
    np.testing.assert_array_equal(
        results[2].reference_distribution.samples, results[3].reference_distribution.samples[:10]
    )
    np.testing.assert_array_equal(
        results[2].generated_distribution.samples, results[3].generated_distribution.samples[:10]
    )


def test_determinism_same_seed():
    real, generated = _frame(200, 0, 1), _frame(200, 0.2, 2)
    cfg = make_options(uncertainty="both", n_replicates=10, seed=7)
    a, b = [Estimator(MeanDiff(), cfg).run(real, generated, SCHEMA)["score"] for _ in range(2)]
    assert (a.value, a.ci_low, a.ci_high, a.ideal) == (b.value, b.ci_low, b.ci_high, b.ideal)
    np.testing.assert_array_equal(
        a.generated_distribution.samples, b.generated_distribution.samples
    )


def test_different_seeds_change_sampled_distributions():
    real, generated = _frame(200, 0, 1), _frame(200, 0.2, 2)
    a, b = [
        Estimator(
            MeanDiff(),
            make_options(uncertainty="both", n_replicates=10, seed=seed),
        ).run(real, generated, SCHEMA)["score"]
        for seed in [1, 2]
    ]
    assert not np.array_equal(a.generated_distribution.samples, b.generated_distribution.samples)


def test_fixed_mode_preserves_both_points_without_distributions_or_labels():
    stat = Estimator(MeanDiff(), make_options()).run(_frame(200, 0, 1), _frame(200, 0, 2), SCHEMA)[
        "score"
    ]
    assert np.isfinite(stat.value) and np.isfinite(stat.ideal)
    assert stat.generated_distribution is stat.reference_distribution is None
    assert not hasattr(stat, "comparison")
    assert np.isnan(stat.ci_low) and np.isnan(stat.ideal_ci_low)
    assert DiagnosticFlag.NO_CI_REQUESTED in stat.flags


def test_insufficient_data_blocks():
    stat = Estimator(MeanDiff(), make_options()).run(_frame(2, 0, 1), _frame(2, 0, 2), SCHEMA)[
        "score"
    ]
    assert DiagnosticFlag.INSUFFICIENT_DATA in stat.flags
    assert np.isnan(stat.value) and np.isnan(stat.ideal)


def test_small_computable_sample_keeps_both_distributions():
    stat = Estimator(MeanDiff(), make_options(uncertainty="both", n_replicates=10)).run(
        _frame(20, 0, 1), _frame(20, 0, 2), SCHEMA
    )["score"]
    assert stat.n_eff == 10
    assert np.isfinite([stat.ci_low, stat.ci_high, stat.ideal_ci_low, stat.ideal_ci_high]).all()
    assert (
        stat.generated_distribution.finite_count == stat.reference_distribution.finite_count == 10
    )


def test_failed_draws_preserve_valid_reference_and_finite_summary():
    class Intermittent(MeanDiff):
        def __init__(self):
            super().__init__()
            self.generated_calls = 0

        def compute(self, reference, candidate, schema, *, rng):
            if candidate.x.min() >= 100:
                self.generated_calls += 1
                return {"score": 0.7 if self.generated_calls % 3 == 0 else np.nan}
            return {"score": 0.9}

    # The synthetic failure schedule counts calls on one metric instance.
    stat = Estimator(
        Intermittent(), make_options(uncertainty="both", n_replicates=10, n_jobs=1)
    ).run(_frame(40, 0, 1), _frame(40, 101, 2), SCHEMA)["score"]
    assert stat.value == 0.7 and stat.ideal == 0.9
    assert stat.generated_distribution.finite_count == 3
    assert stat.generated_distribution.unavailable_count == 7
    assert stat.reference_distribution.finite_count == 10
    assert stat.ci_low == stat.ci_high == 0.7
    assert DiagnosticFlag.NONFINITE_REPLICATES in stat.flags


def test_source_size_disparity_is_visible_without_suppressing_results():
    stat = Estimator(MeanDiff(), make_options(uncertainty="both", n_replicates=10)).run(
        _frame(2000, 0, 1), _frame(50, 0, 2), SCHEMA
    )["score"]
    assert (stat.n_holdout, stat.n_generated, stat.n_eff) == (2000, 50, 25)
    assert np.isfinite([stat.ci_low, stat.ci_high, stat.ideal_ci_low, stat.ideal_ci_high]).all()
