"""Prediction comparisons preserve typed label identity and target units."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.execution.python.agreement import (
    classification_agreement,
    regression_disagreement,
)
from tabgen_eval.preprocessing.categories import CategoryVocabulary

pytestmark = pytest.mark.unit


def test_classification_compares_typed_labels_in_independent_class_orders():
    generated_classes = CategoryVocabulary.fit(pd.Series([1, "1", 1.0], dtype=object))
    reference_classes = CategoryVocabulary.fit(pd.Series([1.0, 1, "1"], dtype=object))
    generated = np.eye(3)
    reference = np.eye(3)[[1, 2, 1]]
    assert classification_agreement(
        generated, reference, generated_classes, reference_classes
    ) == pytest.approx(2 / 3)


def test_regression_difference_retains_squared_target_units():
    assert regression_disagreement(np.array([1.0, 4.0]), np.array([3.0, 5.0])) == 2.5


@pytest.mark.parametrize("function", [classification_agreement, regression_disagreement])
def test_unavailable_predictions_do_not_claim_agreement(function):
    args = (None, None) if function is classification_agreement else ()
    assert np.isnan(function(None, np.array([1.0]), *args))


def test_different_scoring_pool_lengths_are_rejected():
    with pytest.raises(ValueError, match="identical scoring rows"):
        regression_disagreement(np.ones(2), np.ones(3))
