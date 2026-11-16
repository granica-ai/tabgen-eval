"""Retired utility entry points cannot be selected through the public API."""

import inspect

import pandas as pd
import pytest

import tabgen_eval
from tabgen_eval.registry.registry import resolve_metrics
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def test_holdout_api_has_no_explicit_training_roles_or_utility_entry_point():
    assert not hasattr(tabgen_eval, "evaluate_utility")
    parameters = inspect.signature(tabgen_eval.evaluate).parameters
    assert "real_train" not in parameters and "real_test" not in parameters
    metrics = resolve_metrics(["mle"], make_options())
    assert [metric.name for metric in metrics] == ["mle"]


@pytest.mark.parametrize("name", ["prr", "prediction_agreement", "global_mle"])
def test_removed_metrics_are_unregistered_and_rejected(name):
    assert name not in {metric["name"] for metric in tabgen_eval.list_metrics()}
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match=name):
        tabgen_eval.evaluate(
            tabgen_eval.EvalData(
                test=frame, generated=frame, schema=tabgen_eval.Schema(numeric=["x"])
            ),
            metrics=[name],
        )
