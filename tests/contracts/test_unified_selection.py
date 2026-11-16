"""Unified input-role, public selection, and configuration contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib import import_module
from itertools import product

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.data import EvalData
from tabgen_eval.metrics import PRDC
from tabgen_eval.registry import (
    SELECTIONS,
    build_metrics,
    list_groups,
    list_metrics,
    resolve_selection,
)
from tabgen_eval.schema import Schema
from tests.support.options import make_options, public_kwargs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("presence", list(product((False, True), repeat=3)))
def test_eval_data_accepts_exactly_the_four_role_combinations(presence):
    keys = ("train", "test", "generated")
    roles = frozenset(key for key, present in zip(keys, presence, strict=True) if present)
    frames = {role: pd.DataFrame({"x": [1.0]}) for role in roles}
    valid = roles in (
        {"test"},
        {"test", "generated"},
        {"train", "generated"},
        {"train", "test", "generated"},
    )
    if valid:
        assert EvalData(**frames, schema=Schema(numeric=["x"])).roles == roles
    else:
        with pytest.raises(ValueError, match="Valid configurations:.*test.*generated.*train"):
            EvalData(**frames, schema=Schema(numeric=["x"]))


def test_eval_data_is_keyword_only_and_frozen_without_dataframe_equality():
    frame = pd.DataFrame({"x": [1.0]})
    data = EvalData(test=frame, schema=Schema(numeric=["x"]))
    with pytest.raises(TypeError):
        EvalData(frame, schema=Schema(numeric=["x"]))
    with pytest.raises(FrozenInstanceError):
        data.test = frame.copy()
    assert data != EvalData(test=frame, schema=Schema(numeric=["x"]))
    assert data.test is frame
    assert EvalData(test=frame.iloc[:0], schema=Schema(numeric=["x"])).roles == {"test"}


@pytest.mark.parametrize("role", ["train", "test", "generated"])
def test_eval_data_validates_supplied_frame_types(role):
    with pytest.raises(TypeError, match=role):
        EvalData(**{role: np.ones((2, 1))}, schema=Schema(numeric=["x"]))


def test_eval_data_retains_schema_as_the_only_target_location():
    frame = pd.DataFrame({"x": [1.0], "label": [0]})
    schema = Schema(numeric=["x"], categorical=["label"], target="label")
    assert EvalData(test=frame, schema=schema).schema is schema
    with pytest.raises(TypeError, match="schema"):
        EvalData(test=frame, schema={"target": "label"})
    with pytest.raises(TypeError, match="target"):
        EvalData(test=frame, target="label", schema=Schema(numeric=["x"]))


def test_discovery_exposes_atomic_metrics_and_complete_groups():
    names = {entry["name"] for entry in list_metrics()}
    assert names == {
        "shape",
        "trend",
        "precision",
        "recall",
        "density",
        "coverage",
        "cond_shape",
        "cond_trend",
        "recovery",
        "concentration",
        "linear_c2st",
        "nonlinear_c2st",
        "dcr_ratio",
        "near_match_rate",
        "replication_ratio",
        "mle",
    }
    assert {"sanity", "dcr", "prdc", "conditional_fidelity"}.isdisjoint(names)
    assert set(list_groups()) == {
        "fidelity",
        "geometric_fidelity",
        "conditional_fidelity",
        "concentration",
        "detection",
        "memorization",
        "utility",
    }
    assert all(entry["enabled_by_default"] for entry in list_metrics())


@pytest.mark.parametrize("roles", [{"test"}, {"generated", "test"}])
def test_default_selection_includes_all_non_memorization_metrics(roles):
    selected = resolve_selection(roles=roles)
    assert {item.name for item in selected} == {
        item.name for item in SELECTIONS if item.group != "memorization"
    }
    assert "mle" in {item.name for item in selected}
    assert "nonlinear_c2st" in {item.name for item in selected}


def test_training_generated_defaults_include_only_metrics_with_available_roles():
    selected = resolve_selection(roles={"train", "generated"})
    assert [item.name for item in selected] == ["near_match_rate", "replication_ratio"]
    assert resolve_selection(roles={"train", "test", "generated"}) == SELECTIONS


def test_explicit_memorization_group_requires_dcr_denominator_source():
    with pytest.raises(ValueError, match="dcr_ratio requires test"):
        resolve_selection(group=["memorization"], roles={"train", "generated"})
    with pytest.raises(ValueError, match="dcr_ratio requires train"):
        resolve_selection(metrics=["dcr_ratio"], roles={"test", "generated"})
    with pytest.raises(ValueError, match="shape requires test"):
        resolve_selection(metrics=["shape"], roles={"train", "generated"})


def test_metric_filter_overrides_group_compatibility_but_validates_both_names():
    selected = resolve_selection(
        group=["memorization"], metrics=["shape"], roles={"test", "generated"}
    )
    assert [item.name for item in selected] == ["shape"]
    with pytest.raises(ValueError, match="Unknown group.*typo"):
        resolve_selection(group=["typo"], metrics=["shape"], roles={"test"})
    with pytest.raises(ValueError, match="Unknown metric.*prdc"):
        resolve_selection(group=["geometric_fidelity"], metrics=["prdc"], roles={"test"})


def test_selection_deduplicates_in_registry_order_and_includes_companions():
    selected = resolve_selection(
        metrics=["coverage", "cond_shape", "precision", "coverage", "dcr_ratio"],
        roles={"train", "test", "generated"},
    )
    assert [item.name for item in selected] == ["precision", "coverage", "cond_shape", "dcr_ratio"]
    assert selected[2].outputs == ("cond_shape_all", "cond_shape_rare")
    assert selected[3].primary == "ratio"
    assert {"median", "q05", "holdout_q50", "ratio"} <= set(selected[3].outputs)


@pytest.mark.parametrize("kwargs", [{"metrics": "shape"}, {"group": "fidelity"}, {"metrics": [1]}])
def test_filters_require_sequences_of_registered_strings(kwargs):
    with pytest.raises(TypeError):
        resolve_selection(roles={"test"}, **kwargs)


def test_shared_computations_are_instantiated_once_without_mutating_stat_identity():
    selected = resolve_selection(
        metrics=["precision", "coverage", "cond_shape", "recovery"], roles={"test"}
    )
    options = make_options()
    computations = build_metrics(selected, options)
    assert [item.name for item in computations] == ["prdc", "conditional_fidelity"]
    prdc, conditional = computations
    assert prdc.requested_components == {"precision", "coverage"}
    assert conditional.requested_components == {"cond_shape", "recovery"}
    assert prdc.stat_names == ("precision", "recall", "density", "coverage")
    assert prdc.nearest_k == PRDC().nearest_k
    assert not hasattr(prdc, "backend")
    assert "metric_kwargs" not in options


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_uncertainty_dictionary_is_retained_in_public_options(mode):
    from tabgen_eval import evaluate

    uncertainty = None if mode == "none" else {"mode": mode, "n_replicates": np.int64(7)}
    frame = pd.DataFrame({"x": np.linspace(0, 1, 30)})
    report = evaluate(
        EvalData(test=frame, generated=frame, schema=Schema(numeric=["x"])),
        metrics=["shape"],
        seed=[41, 5],
        n_jobs=2,
        uncertainty=uncertainty,
    )
    assert report.options_dict["uncertainty"] == uncertainty
    if uncertainty is not None:
        assert type(report.options_dict["uncertainty"]["n_replicates"]) is int
        assert isinstance(uncertainty["n_replicates"], np.int64)
    assert report.options_dict["seed"] == [41, 5]
    assert report.options_dict["n_jobs"] == 2
    assert "n_replicates" not in report.options_dict
    assert "uq_config" not in report.provenance


def test_separate_uq_argument_is_rejected():
    from tabgen_eval import evaluate

    frame = pd.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(TypeError, match="uq_config"):
        evaluate(
            EvalData(test=frame, schema=Schema(numeric=["x"])), metrics=["shape"], uq_config=None
        )


def test_public_nonutility_selections_do_not_require_a_target():
    from tabgen_eval import evaluate

    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
    options = make_options()
    fidelity = evaluate(
        EvalData(test=frame, generated=frame, schema=Schema(numeric=["x"])),
        metrics=["shape"],
        **public_kwargs(options),
    )
    memory = evaluate(
        EvalData(train=frame, generated=frame, schema=Schema(numeric=["x"])),
        **public_kwargs(options),
    )
    assert np.isfinite(fidelity["shape"].value)
    assert [metric.name for metric in memory.results] == ["near_match_rate", "replication_ratio"]


def test_declared_but_absent_target_is_rejected_before_computation(monkeypatch):
    from tabgen_eval import evaluate
    from tabgen_eval.models.xgboost import XGBoostModel

    monkeypatch.setattr(
        XGBoostModel, "load_backend", lambda: pytest.fail("dependency loaded before target check")
    )
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
    data = EvalData(test=frame, generated=frame, schema=Schema(numeric=["x"], target="absent"))
    with pytest.raises(ValueError, match="Schema declares columns not present.*absent"):
        evaluate(data, metrics=["mle"], uncertainty=None)


def test_private_memorization_rejects_incompatible_metric_budgets():
    from tabgen_eval.metrics import DCR, NearMatchRate
    from tests.support.evaluation import memorization_report as evaluate_memorization

    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
    with pytest.raises(ValueError, match="same max_samples"):
        evaluate_memorization(
            frame,
            frame,
            Schema(numeric=["x"]),
            holdout=frame,
            metrics=[DCR(max_samples=10), NearMatchRate(max_samples=20)],
            options=make_options(),
        )


def test_training_schema_and_memory_score_are_independent_of_other_selected_groups():
    from tabgen_eval import evaluate

    # The caller accepts inferred training roles once, then all procedures use them.
    train = pd.DataFrame({"x": np.arange(20, dtype=float)})
    data = EvalData(
        train=train, test=train + 0.001, generated=train + 0.000001, schema=Schema.infer(train)
    )
    options = make_options(seed=42)
    memory = evaluate(data, metrics=["near_match_rate"], **public_kwargs(options))
    combined = evaluate(data, metrics=["shape", "near_match_rate"], **public_kwargs(options))
    assert memory["near_match_rate"].value == combined["near_match_rate"].value == 0.0
    assert memory.schema_dict["categorical"] == ["x"]
    procedures = combined.provenance["procedures"]
    assert procedures["memorization"]["schema"] == memory.schema_dict
    assert procedures["comparison"]["schema"] == memory.schema_dict


@pytest.mark.parametrize("with_test", [False, True])
def test_details_requires_a_selected_generated_test_comparison(with_test, monkeypatch):
    module = import_module("tabgen_eval.evaluate")
    monkeypatch.setattr(
        module, "evaluate_memorization", lambda *a, **k: pytest.fail("computed unsupported details")
    )
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
    data = EvalData(
        train=frame,
        test=frame if with_test else None,
        generated=frame,
        schema=Schema(numeric=["x"]),
    )
    with pytest.raises(ValueError, match="generated/test point comparison"):
        module.evaluate(data, metrics=["near_match_rate"], details=True)
