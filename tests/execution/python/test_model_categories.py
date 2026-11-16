"""Typed category identities and training-only vocabularies in fitted models."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.xgboost.utility import compute_mle
from tabgen_eval.models import XGBoostModel
from tabgen_eval.preprocessing.categories import MISSING_CATEGORY
from tabgen_eval.preprocessing.frames import fold_design
from tabgen_eval.preprocessing.predictors import PredictorPreprocessor, predictor_design
from tests.support.evaluation import paired_report as evaluate
from tests.support.linear_c2st import compute_partitioned as linear_reference
from tests.support.options import make_options
from tests.support.reports import without_memory_plans

pytestmark = pytest.mark.unit


def categorical_frame(values):
    return pd.DataFrame({"c": pd.Series(values, dtype=object)})


@pytest.mark.parametrize("design", ["classifier", "utility", "retention"])
def test_predictor_categories_preserve_types_strings_and_missing_states(design):
    labels = [1, 1.0, "1", True, " A ", "A", "1.0", "nan", "<MISSING>", None]
    train = categorical_frame(labels + [np.int64(1), np.float32(1.0)])
    scoring = categorical_frame(labels + ["unseen", np.int32(1), np.float64(1.0)])
    schema = Schema(categorical=["c"])
    if design == "classifier":
        training, heldout, _, _ = fold_design(
            train, train.copy(), scoring, scoring.copy(), schema, 2**30
        )
    elif design == "utility":
        training, heldout = predictor_design(train, scoring, schema, 2**30)
    else:
        encoding = PredictorPreprocessor.fit(train, schema, 2**30)
        training, heldout = encoding.transform(train).numpy(), encoding.transform(scoring).numpy()
    assert training.shape[1] == heldout.shape[1] == (10 if design == "classifier" else 1)
    assert len(np.unique(training[:10], axis=0)) == 10
    np.testing.assert_equal(training[0], training[10])
    np.testing.assert_equal(training[1], training[11])
    np.testing.assert_equal(heldout[:10], training[:10])
    np.testing.assert_equal(heldout[11], training[0])
    np.testing.assert_equal(heldout[12], training[1])
    if design == "classifier":
        assert not heldout[10].any()
    else:
        assert heldout[10, 0] == 10


@pytest.mark.parametrize("design", ["classifier", "utility", "retention"])
def test_unobserved_missing_and_scoring_labels_never_expand_training_vocab(design):
    train = categorical_frame(["a", "b", "a", "b"])
    scoring = categorical_frame([None, "new", " A ", "nan"])
    changed = categorical_frame([1, True, 1.0, "1"])
    schema = Schema(categorical=["c"])
    if design == "classifier":
        first, heldout, _, _ = fold_design(train, train, scoring, scoring, schema, 2**30)
        second, _, _, _ = fold_design(train, train, changed, changed, schema, 2**30)
    else:
        first, heldout = predictor_design(train, scoring, schema, 2**30)
        second, _ = predictor_design(train, changed, schema, 2**30)
    np.testing.assert_equal(first, second)
    assert first.shape[1] == heldout.shape[1] == (2 if design == "classifier" else 1)
    np.testing.assert_equal(heldout, np.tile(heldout[0], (len(heldout), 1)))


@pytest.mark.parametrize("design", ["classifier", "utility", "retention"])
def test_explicit_missing_category_is_distinct_from_actual_null_and_literal_label(design):
    train = categorical_frame([MISSING_CATEGORY, "<MISSING>", None, "nan"] * 3)
    schema = Schema(categorical=["c"])
    if design == "classifier":
        features, _, _, _ = fold_design(train, train, train, train, schema, 2**30)
    else:
        features, _ = predictor_design(train, train, schema, 2**30)
    assert features.shape[1] == (4 if design == "classifier" else 1)
    assert len(np.unique(features[:4], axis=0)) == 4


def classifier_stub(monkeypatch):
    calls = []
    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: None))

    def fit(self, train, labels, **kwargs):
        train = train.detach().cpu().numpy()
        self._stub_index = len(calls)
        calls.append((train.copy(), labels.copy(), None))
        self.diagnostics = {"converged": True}
        return self

    def predict(self, scoring, **kwargs):
        scoring = scoring.detach().cpu().numpy()
        train, labels, _ = calls[self._stub_index]
        calls[self._stub_index] = train, labels, scoring.copy()
        mapping = dict(zip(train[:, 0], labels, strict=True))
        predictions = np.zeros((len(scoring), len(np.unique(labels))))
        for row, value in enumerate(scoring[:, 0]):
            predictions[row, mapping[value]] = 1
        return predictions

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    return calls


def utility_outcome(reference, candidate, baseline):
    return compute_mle(
        reference,
        candidate,
        baseline,
        Schema(numeric=["x"], categorical=["y"], target="y"),
        task="classification",
        device="cpu",
        memory_limit_mb=1024,
        rng=np.random.default_rng(5),
    )


@pytest.mark.parametrize(
    "labels",
    [[1, 1.0, "1", True], ["A", " A ", "1", "1.0"], ["nan", "NaN", "<MISSING>", ""]],
)
def test_classification_targets_keep_exact_identities_and_exclude_actual_nulls(monkeypatch, labels):
    calls = classifier_stub(monkeypatch)
    frame = pd.DataFrame(
        {
            "x": np.tile(np.arange(len(labels) + 1), 2),
            "y": pd.Series((labels + [None]) * 2, dtype=object),
        }
    )
    outcome = utility_outcome(frame.iloc[::-1], frame, frame.copy())
    assert outcome.generated_score == outcome.reference_score == outcome.ratio == 1.0
    for train, targets, scoring in calls:
        assert len(np.unique(targets)) == len(labels)
        assert len(train) == len(scoring) == len(labels) * 2
    for info in outcome.diagnostics["fits"].values():
        assert len(info["classes"]) == len(labels)
    json.dumps(outcome.diagnostics, allow_nan=False)


def test_target_support_check_distinguishes_integer_from_string_labels(monkeypatch):
    calls = classifier_stub(monkeypatch)
    reference = pd.DataFrame({"x": [0, 1, 0, 1], "y": pd.Series([1, 2, 1, 2], dtype=object)})
    candidate = reference.assign(y=pd.Series(["1", "2", "1", "2"], dtype=object))
    outcome = utility_outcome(reference, candidate, reference.copy())
    assert np.isnan(outcome.generated_score)
    assert outcome.diagnostics["fits"]["generated"]["reason"] == "classification_class_support"
    assert len(calls) == 2
    assert outcome.reference_score == 1.0


def test_mixed_target_diagnostics_and_paired_replicates_preserve_identities():
    pytest.importorskip("xgboost", minversion="2.1")
    rng = np.random.default_rng(31)
    labels = [np.int64(1), np.float64(1.0), np.str_("1"), np.bool_(True)]
    frames = []
    for _ in range(2):
        indices = np.arange(256) % len(labels)
        target = pd.Series([labels[i] for i in indices], dtype=object)
        target.iloc[::19] = None
        frame = pd.DataFrame(
            {
                "x": indices + rng.normal(scale=0.05, size=len(indices)),
                "c": pd.Series(([None, "<MISSING>", "nan", " A "] * 64), dtype=object),
                "y": target,
            }
        )
        frames.append(frame)
    schema = Schema(numeric=["x"], categorical=["c", "y"], target="y")
    reports = [
        evaluate(
            *frames,
            schema,
            metrics=["mle"],
            options=make_options(
                n_jobs=jobs,
                seed=17,
                uncertainty="both",
                n_replicates=2,
            ),
        )
        for jobs in [1, 2]
    ]
    diagnostics = [report.results[0].diagnostics for report in reports]
    np.testing.assert_equal(
        without_memory_plans(diagnostics[0]), without_memory_plans(diagnostics[1])
    )
    expected = {(int, 1), (float, 1.0), (str, "1"), (bool, True)}
    for report in reports:
        info = report.results[0].diagnostics
        assert np.isfinite(report["mle/ratio"].value)
        assert report["mle/ratio"].generated_distribution.finite_count == 2
        for outcome in [info["point"], *info["replicates"]]:
            for fit in outcome["diagnostics"]["fits"].values():
                assert {(type(value), value) for value in fit["classes"]} == expected
            assert all(count < 128 for count in outcome["diagnostics"]["usable_rows"].values())
            exported = json.loads(json.dumps(outcome["diagnostics"], allow_nan=False))
            assert exported == outcome["diagnostics"]
    # Source missingness remains available to the caller after preparation.
    assert frames[0]["c"].isna().sum() == 64
    assert frames[0]["y"].isna().sum() == 14


def test_cpu_and_python_classifier_fit_same_typed_feature_space():
    pytest.importorskip("torch")
    from tabgen_eval.execution.runtime.dispatch import context_for
    from tabgen_eval.metrics import LinearC2ST

    rng = np.random.default_rng(12)
    labels = [1, 1.0, "1", True, " A ", "A", "nan", "<MISSING>", None, MISSING_CATEGORY]
    frames = []
    for shift in [0.0, 0.4]:
        frame = pd.DataFrame(
            {
                "x": rng.normal(loc=shift, size=240),
                "c": pd.Series(
                    [labels[i] for i in rng.integers(len(labels), size=240)], dtype=object
                ),
            }
        )
        frames.append(frame)
    frames[1].loc[::17, "c"] = "candidate-only"
    schema = Schema(numeric=["x"], categorical=["c"])
    folds = [np.arange(len(frame)) % 3 for frame in frames]
    metric = LinearC2ST(device="cpu", solver="direct", tolerance=1e-07, ridge=0.03)
    oracle = linear_reference(metric, *frames, schema, *folds, seed=6, point=True)
    context = context_for(metric, *frames, schema)
    native = context.compute_classifier(
        metric,
        np.arange(240)[None],
        np.arange(240)[None],
        folds[0][None],
        folds[1][None],
        [6],
    )[0]
    assert native.converged and oracle.converged
    assert native.value == pytest.approx(oracle.value, abs=2e-4)
    assert [fold["features"] for fold in native.folds] == [
        fold["features"] for fold in oracle.folds
    ]
    assert {fold["features"] for fold in native.folds} == {12}  # Includes generated-only category.
