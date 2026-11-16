"""Raw predictive scores and ratios retain each task's direct interpretation."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from tabgen_eval.execution.python.utility import regression_mae
from tabgen_eval.execution.xgboost.utility import (
    MLEOutcome,
    classification_ratio,
    regression_ratio,
)

pytestmark = pytest.mark.unit
ZERO_SCORES = "zero generated and reference scores"


@pytest.mark.parametrize(
    "generated,reference,ratio",
    [
        (1.0, 0.5, 2.0),
        (0.25, 0.5, 0.5),
        (0.0, 0.5, 0.0),
        (4.0, 2.0, 2.0),
        (0.5, 0.0, np.inf),
        (0.5, -0.0, np.inf),
    ],
)
def test_regression_ratio_divides_raw_mean_absolute_errors(generated, reference, ratio):
    outcome = regression_ratio(generated, reference)
    assert outcome.generated_score == generated
    assert outcome.reference_score == reference
    assert outcome.ratio == ratio
    assert outcome.reason is None


@pytest.mark.parametrize("ratio_function", [regression_ratio, classification_ratio])
@pytest.mark.parametrize("generated,reference", [(0.0, 0.0), (-0.0, -0.0)])
def test_zero_divided_by_zero_is_undefined_with_an_explicit_reason(
    ratio_function, generated, reference
):
    outcome = ratio_function(generated, reference)
    assert np.isnan(outcome.ratio)
    assert outcome.generated_score == generated
    assert outcome.reference_score == reference
    assert outcome.reason == ZERO_SCORES


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -0.01, -4.0])
@pytest.mark.parametrize("side", ["generated", "reference"])
def test_invalid_mae_is_preserved_and_distinct_from_valid_zero_scores(invalid, side):
    generated, reference = (invalid, 0.5) if side == "generated" else (0.5, invalid)
    outcome = regression_ratio(generated, reference)
    np.testing.assert_equal(outcome.generated_score, generated)
    np.testing.assert_equal(outcome.reference_score, reference)
    assert np.isnan(outcome.ratio)
    assert outcome.reason == "undefined regression MAE"
    assert outcome.reason != ZERO_SCORES


@pytest.mark.parametrize("ratio_function", [regression_ratio, classification_ratio])
def test_tiny_positive_reference_is_not_smoothed_or_capped(ratio_function):
    tiny = np.nextafter(0.0, 1.0)
    assert ratio_function(2 * tiny, tiny).ratio == 2.0
    assert ratio_function(1e-10, 1e-16).ratio == pytest.approx(1e6)
    assert ratio_function(0.0, tiny).ratio == 0.0
    assert ratio_function(tiny, 0.0).ratio == np.inf


@pytest.mark.parametrize(
    "generated,reference,ratio",
    [
        (0.7, 0.9, 7 / 9),
        (0.9, 0.7, 9 / 7),
        (1.0, 0.75, 4 / 3),
        (0.75, 0.5, 1.5),
        (0.75, 0.0, np.inf),
        (0.5, 0.75, 2 / 3),
        (0.0, 0.75, 0.0),
        (0.5, 0.5, 1.0),
        (0.1, 0.4, 0.25),
        (0.4, 0.1, 4.0),
    ],
)
def test_classification_divides_raw_aurocs_including_chance_and_below_chance(
    generated, reference, ratio
):
    outcome = classification_ratio(generated, reference)
    assert outcome.generated_score == generated
    assert outcome.reference_score == reference
    assert outcome.ratio == pytest.approx(ratio)
    assert outcome.reason is None


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -0.01, 1.01])
@pytest.mark.parametrize("side", ["generated", "reference"])
def test_invalid_classification_scores_have_an_explicit_reason(invalid, side):
    generated, reference = (invalid, 0.8) if side == "generated" else (0.8, invalid)
    outcome = classification_ratio(generated, reference)
    np.testing.assert_equal(outcome.generated_score, generated)
    np.testing.assert_equal(outcome.reference_score, reference)
    assert np.isnan(outcome.ratio)
    assert outcome.reason == "undefined classification AUROC"
    assert outcome.reason != ZERO_SCORES


@pytest.mark.parametrize("generated,reference", [(0.5, 0.5), (0.7, 0.9), (0.9, 0.0), (0, 0.7)])
def test_tasks_share_one_raw_score_ratio_policy(generated, reference):
    classification = classification_ratio(generated, reference)
    regression = regression_ratio(generated, reference)
    np.testing.assert_equal(classification.ratio, regression.ratio)
    assert classification.reason == regression.reason


def test_retention_outcome_contains_no_transformed_score_fields():
    names = {field.name for field in fields(MLEOutcome)}
    assert "generated_skill" not in names
    assert "reference_skill" not in names


@pytest.mark.parametrize(
    "prediction,expected",
    [
        ([-1.0, 0.0, 1.0], 0.0),
        ([-1.0, 0.0, 0.0], 1 / 3),
        ([0.0, 0.0, 0.0], 2 / 3),
        ([1.0, 0.0, 1.0], 2 / 3),
    ],
)
def test_mae_matches_independent_absolute_residual_oracle(prediction, expected):
    truth = np.array([-1.0, 0.0, 1.0])
    prediction = np.array(prediction)
    oracle = np.sum(np.abs(truth - prediction)) / len(truth)
    assert oracle == expected
    assert regression_mae(truth, prediction) == expected


@pytest.mark.parametrize("scale,offset", [(0.5, -4.0), (-3.0, 17.0), (4.0, 0.0)])
def test_mae_scales_in_target_units_and_its_ratio_is_unit_invariant(scale, offset):
    truth = np.array([-1.0, 0.0, 1.0])
    generated = np.array([1.0, 0.0, 1.0])
    reference = np.array([-1.0, 1.0, 1.0])
    scores = [regression_mae(truth, prediction) for prediction in (generated, reference)]
    changed = [
        regression_mae(scale * truth + offset, scale * prediction + offset)
        for prediction in (generated, reference)
    ]
    np.testing.assert_allclose(changed, np.abs(scale) * np.asarray(scores), atol=1e-14)
    assert regression_ratio(*changed).ratio == pytest.approx(
        regression_ratio(*scores).ratio, abs=1e-14
    )


@pytest.mark.parametrize(
    "truth,prediction,expected",
    [
        ([1.0], [1.0], 0.0),
        ([1.0], [3.0], 2.0),
        ([2.0, 2.0], [2.0, 2.0], 0.0),
        ([2.0, 2.0], [1.0, 1.0], 1.0),
    ],
)
def test_one_row_and_constant_targets_have_defined_mae(truth, prediction, expected):
    assert regression_mae(truth, prediction) == expected


@pytest.mark.parametrize(
    "truth,prediction",
    [
        ([], []),
        ([0.0, np.nan], [0.0, 1.0]),
        ([0.0, np.inf], [0.0, 1.0]),
        ([0.0, 1.0], [np.nan, 1.0]),
        ([0.0, 1.0], [np.inf, 1.0]),
        ([0.0, 1.0], [0.0]),
        ([[0.0, 1.0]], [[0.0, 1.0]]),
        (1.0, 1.0),
    ],
)
def test_undefined_mae_remains_unavailable(truth, prediction):
    assert np.isnan(regression_mae(truth, prediction))
