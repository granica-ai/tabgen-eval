"""EvalData announces role-compatible defaults without running evaluation work."""

from __future__ import annotations

from importlib import import_module

import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema

pytestmark = pytest.mark.unit

_COMPARISON_METRICS = (
    "  fidelity: shape, trend\n"
    "  geometric_fidelity: precision, recall, density, coverage\n"
    "  conditional_fidelity: cond_shape, cond_trend, recovery\n"
    "  concentration: concentration\n"
    "  detection: linear_c2st, nonlinear_c2st\n"
)
_UTILITY_PREREQUISITE = (
    "  mle requires schema.target; supply it or filter out utility when evaluating.\n"
)


@pytest.mark.parametrize(
    ("roles", "expected_metrics"),
    [
        (("test",), _COMPARISON_METRICS + "  utility: mle\n"),
        (("test", "generated"), _COMPARISON_METRICS + "  utility: mle\n"),
        (
            ("train", "generated"),
            "  memorization: near_match_rate, replication_ratio\n",
        ),
        (
            ("train", "test", "generated"),
            _COMPARISON_METRICS
            + "  memorization: dcr_ratio, near_match_rate, replication_ratio\n"
            + "  utility: mle\n",
        ),
    ],
)
def test_constructor_prints_exact_default_metrics_for_each_valid_configuration(
    roles, expected_metrics, capsys
):
    frame = pd.DataFrame({"x": [1.0]})
    data = EvalData(**dict.fromkeys(roles, frame), schema=Schema(numeric=["x"]))

    expected = f"EvalData ({' + '.join(roles)}) — default evaluate metrics:\n" + expected_metrics
    if "test" in roles:
        expected += _UTILITY_PREREQUISITE
    assert capsys.readouterr().out == expected
    assert data.roles == set(roles)
    repr(data)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("schema", [Schema(numeric=["x"])])
def test_missing_target_is_explained_without_rejecting_filtered_use(schema, capsys):
    EvalData(test=pd.DataFrame({"x": [1.0]}), schema=schema)

    assert capsys.readouterr().out.endswith(_UTILITY_PREREQUISITE)


def test_declared_target_removes_the_utility_prerequisite_note(capsys):
    frame = pd.DataFrame({"x": [1.0], "label": [0]})
    EvalData(test=frame, schema=Schema(numeric=["x"], categorical=["label"], target="label"))

    summary = capsys.readouterr().out
    assert "  utility: mle\n" in summary
    assert "requires schema.target" not in summary


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"train": pd.DataFrame()},
        {"generated": pd.DataFrame()},
        {"train": pd.DataFrame(), "test": pd.DataFrame()},
        {"test": [[1.0]]},
        {"test": pd.DataFrame(), "schema": {}},
    ],
)
def test_invalid_constructor_inputs_raise_without_printing_a_summary(kwargs, capsys):
    with pytest.raises((ValueError, TypeError)):
        EvalData(**kwargs, schema=Schema(numeric=["x"]))

    assert capsys.readouterr().out == ""


def test_constructor_does_not_infer_prepare_or_fit_data_or_load_optional_engines(
    monkeypatch, capsys
):
    evaluation = import_module("tabgen_eval.evaluate")
    registry = import_module("tabgen_eval.registry")
    schema = import_module("tabgen_eval.schema")

    def unexpected(*args, **kwargs):
        pytest.fail("EvalData construction must only validate roles and describe metric defaults")

    monkeypatch.setattr(registry, "build_metrics", unexpected)
    monkeypatch.setattr(evaluation, "build_metrics", unexpected)
    monkeypatch.setattr(schema, "infer_schema", unexpected)
    monkeypatch.setattr(Schema, "infer", unexpected)
    monkeypatch.setattr(evaluation.XGBoostModel, "load_backend", unexpected)
    # An empty frame with no columns is still a supplied role. Its eventual
    # metric applicability is evaluation's concern, not constructor work.
    frame = pd.DataFrame()
    data = EvalData(train=frame, test=frame, generated=frame, schema=Schema(numeric=["x"]))

    assert data.train is frame
    assert data.test is frame
    assert data.generated is frame
    assert capsys.readouterr().out.count("default evaluate metrics:") == 1
