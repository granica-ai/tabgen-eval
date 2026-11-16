"""Fixed detector policies, symmetric precision and held-out independence."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import LinearC2ST, NonLinearC2ST
from tabgen_eval.preprocessing import fit_fold, prepare_sources
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options


def test_classifiers_share_prepared_sources_with_compatible_metric_inputs():
    from tabgen_eval.execution.runtime.dispatch import ExecutionSession
    from tabgen_eval.metrics import Shape

    real = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["b", "a", "b"]})
    schema = Schema(numeric=["x"], categorical=["c"])
    session = ExecutionSession(real, real.copy(), schema, classifier_preprocessing=True)
    shape = session.context_for(Shape(device="cpu", dtype="float32"))
    linear = session.context_for(LinearC2ST(device="cpu", dtype="float32"))
    nonlinear = session.context_for(NonLinearC2ST(device="cpu", dtype="float32"))
    assert shape.reference is linear.reference is nonlinear.reference
    assert shape.candidate is linear.candidate is nonlinear.candidate


def test_classifier_precision_matching_does_not_round_other_metric_inputs():
    from tabgen_eval.execution.runtime.dispatch import ExecutionSession
    from tabgen_eval.metrics import Shape

    real = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    fake = real.astype("float32")
    session = ExecutionSession(real, fake, Schema(numeric=["x"]), classifier_preprocessing=True)
    shape = session.context_for(Shape(device="cpu", dtype="float64"))
    linear = session.context_for(LinearC2ST(device="cpu", dtype="float64"))
    np.testing.assert_array_equal(shape.reference.numeric[:, 0], real.x)
    np.testing.assert_array_equal(linear.reference.numeric[:, 0], fake.x)
    assert shape.reference is not linear.reference


def test_unlabeled_prepared_storage_recovers_classifier_dictionary_without_new_features():
    from tabgen_eval.preprocessing.columns import PreparedColumns

    real = pd.DataFrame({"c": ["b", "a", None]})
    fake = pd.DataFrame({"c": ["c", "b", None]})
    schema = Schema(categorical=["c"])
    columns = PreparedColumns.prepare(real, fake, schema)
    assert columns.category_values is None
    sources = prepare_sources(real, fake, schema, columns=columns)
    assert sources.category_values == (("a", "b", "c"),)


@pytest.mark.parametrize("metric", [LinearC2ST, NonLinearC2ST])
@pytest.mark.parametrize("value", ["standard", "robust", "none"])
def test_scaling_is_a_fixed_definition_not_an_override(metric, value):
    with pytest.raises(TypeError, match="numeric_scaling"):
        metric(numeric_scaling=value)


def test_matching_observation_precision_is_symmetric_and_preserves_sources():
    real = pd.DataFrame(
        {
            "x": np.array([0.1, 0.2, 0.3], dtype="float64"),
            "y": np.array([0.3, 0.4, 0.5], dtype="float32"),
        }
    )
    fake = pd.DataFrame({"x": real.x.astype("float32"), "y": real.y.astype("float64")})
    old = real.copy(deep=True)
    sources = prepare_sources(real, fake, Schema(numeric=["x", "y"]), dtype="float64")
    np.testing.assert_array_equal(sources.reference.numeric, sources.candidate.numeric)
    assert sources.precision["observation_bits"] == {"x": 32, "y": 32}
    assert sources.reference.numeric.dtype.is_floating_point
    pd.testing.assert_frame_equal(real, old)


@pytest.mark.parametrize("policy", ["linear", "xgboost"])
def test_heldout_values_cannot_reorder_or_create_learned_features(policy):
    # A held-out first appearance of a generated training category must not
    # influence even its feature ordering. Storage IDs themselves may differ.
    real = pd.DataFrame({"x": [100.0, 1.0, 3.0, 200.0], "c": ["b", "a", "a", "outside"]})
    fake = pd.DataFrame({"x": [300.0, 2.0, 4.0, 400.0], "c": ["outside", "b", "c", "outside"]})
    labels = np.array([0, 1, 1, 0])
    schema = Schema(numeric=["x"], categorical=["c"])

    def prepared(r):
        source = prepare_sources(r, fake, schema)
        return fit_fold(
            source.reference,
            source.candidate,
            labels,
            labels,
            0,
            cardinalities=source.cardinalities,
            numeric_names=source.numeric_names,
            categorical_names=source.categorical_names,
            category_values=source.category_values,
            policy=policy,
        )

    before = prepared(real)
    after = prepared(real.assign(x=[-1e30, 1.0, 3.0, 1e30], c=["new", "a", "a", "another-new"]))
    np.testing.assert_array_equal(before.train, after.train)
    assert before.feature_names == after.feature_names
    assert before.categorical_cardinalities == after.categorical_cardinalities == (3,)
    assert len(before.feature_names) == (4 if policy == "linear" else 2)


def test_all_missing_real_and_finite_generated_retains_missingness_signal():
    real = pd.DataFrame({"x": [np.nan] * 12})
    fake = pd.DataFrame({"x": [5.0] * 12})
    report = evaluate(
        real,
        fake,
        Schema(numeric=["x"]),
        metrics=["linear_c2st"],
        options=make_options(device="cpu"),
    )
    assert report["linear_c2st/auc"].value == 1.0
    folds = report.provenance["classifier_diagnostics"]["linear_c2st"]["point"]["generated"][
        "folds"
    ]
    assert all(fold["preprocessing"]["missing_indicators"] == ["x"] for fold in folds)


def test_target_marker_alone_does_not_select_a_classifier_feature():
    real = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "a", "a"]})
    fake = real.assign(label="b")
    selected = prepare_sources(real, fake, Schema(numeric=["x"], target="label"))
    assert selected.numeric_names == ("x",)
    assert selected.categorical_names == ()


def test_classifier_sources_reject_an_independent_third_pool():
    from dataclasses import replace

    from tabgen_eval.preprocessing.columns import PreparedColumns

    real = pd.DataFrame({"x": [1.0, 2.0], "c": ["z", "a"]})
    schema = Schema(numeric=["x"], categorical=["c"])
    columns = PreparedColumns.prepare(real, real.copy(), schema)
    columns = replace(columns, holdout=columns.reference)
    with pytest.raises(ValueError, match="two pools"):
        prepare_sources(real, real.copy(), schema, columns=columns)
