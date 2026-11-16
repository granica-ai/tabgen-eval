"""Raw reference summaries preserve finite observations without recentering."""

import math

import numpy as np
import pytest

from tabgen_eval.statistics.distributions import ScoreDistribution

pytestmark = pytest.mark.unit


def test_summary_owns_samples_and_counts_every_nonfinite_draw():
    samples = np.array([0.0, 0.25, 0.75, 1.0, np.nan, np.inf])
    result = ScoreDistribution.from_samples(samples, confidence_interval=0.8)
    samples[0] = 99
    assert result.samples[0] == 0
    assert result.samples.flags.writeable is False
    assert result.finite_count == 4
    assert result.unavailable_count == 2
    assert result.mean == result.median == 0.5
    assert result.band_low == pytest.approx(0.075)
    assert result.band_high == pytest.approx(0.925)
    assert math.isnan(result.samples[4])
    assert result.samples[5] == np.inf


def test_unavailable_summary_does_not_invent_a_score():
    result = ScoreDistribution.from_samples([np.nan, -np.inf], confidence_interval=0.95)
    assert result.finite_count == 0
    assert result.unavailable_count == 2
    assert all(
        math.isnan(value)
        for value in [result.mean, result.median, result.band_low, result.band_high]
    )


@pytest.mark.parametrize("confidence_interval", [True, "0.05", None, 0, 1, np.nan, np.inf])
def test_summary_rejects_invalid_band_levels(confidence_interval):
    with pytest.raises(ValueError, match="confidence_interval"):
        ScoreDistribution.from_samples([0.5], confidence_interval=confidence_interval)


def test_summary_requires_a_score_vector():
    with pytest.raises(ValueError, match="one-dimensional"):
        ScoreDistribution.from_samples([[0.1, 0.2]], confidence_interval=0.95)


def test_large_finite_scores_do_not_overflow_summary_arithmetic():
    result = ScoreDistribution.from_samples([1e308, 1e308], confidence_interval=0.95)
    assert result.mean == result.median == 1e308
    assert result.band_low == result.band_high == 1e308
    opposite = ScoreDistribution.from_samples([-1e308, 1e308], confidence_interval=0.95)
    assert opposite.mean == opposite.median == 0
    assert opposite.band_low == pytest.approx(-9.5e307)
    assert opposite.band_high == pytest.approx(9.5e307)


def test_nonfinite_reasons_and_requested_counts_preserve_raw_order():
    result = ScoreDistribution.from_samples(
        [0, np.nan, 2, np.inf, -np.inf], confidence_interval=0.95
    )
    assert result.requested_count == 5
    assert result.discarded_reasons == {"nan": 1, "positive_infinity": 1, "negative_infinity": 1}
    assert result.mean == 1
    assert result.reason is None
    bad = ScoreDistribution.from_samples([np.nan], confidence_interval=0.95)
    assert bad.reason == "no finite replicate scores"


def test_explicit_discard_reasons_must_account_for_excluded_draws():
    result = ScoreDistribution.from_samples(
        [0, np.nan], confidence_interval=0.95, discarded_reasons={"undefined_statistic": 1}
    )
    assert result.discarded_reasons == {"undefined_statistic": 1}
    with pytest.raises(ValueError, match="match"):
        ScoreDistribution.from_samples([0, np.nan], confidence_interval=0.95, discarded_reasons={})
    with pytest.raises(ValueError, match="nonnegative"):
        ScoreDistribution.from_samples(
            [np.nan], confidence_interval=0.95, discarded_reasons={"failure": True}
        )


@pytest.mark.parametrize(
    "confidence_interval, expected",
    [
        (0.5, (0.25, 0.75)),
        (0.8, (0.1, 0.9)),
        (0.95, (0.025, 0.975)),
    ],
)
def test_confidence_interval_stores_central_coverage(confidence_interval, expected):
    result = ScoreDistribution.from_samples([0.0, 1.0], confidence_interval=confidence_interval)
    assert result.confidence_interval == confidence_interval
    assert (result.band_low, result.band_high) == pytest.approx(expected)
    assert not hasattr(result, "alpha")
