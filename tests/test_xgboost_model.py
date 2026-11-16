"""Public model lifecycle, prediction modes, and fitted-metric failure boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from tabgen_eval.models.errors import FitUnavailable
from tabgen_eval.models.xgboost import XGBoostModel

pytestmark = pytest.mark.unit


@pytest.fixture
def xgb():
    return pytest.importorskip("xgboost", minversion="2.1")


@pytest.mark.parametrize(
    ("task", "prediction"),
    [("classification", "margin"), ("classification", "probability"), ("regression", "value")],
)
def test_fit_and_repeated_predict_match_independent_engine(xgb, task, prediction, monkeypatch):
    rng = np.random.default_rng(9)
    train, test = rng.normal(size=(120, 3)), rng.normal(size=(30, 3))
    labels = (train[:, 0] > 0).astype(int) if task == "classification" else train[:, 0] * 2
    cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
    expected = cls(device="cpu", tree_method="hist", n_jobs=2, random_state=9)
    expected.fit(train, labels)
    wrapper = XGBoostModel(task=task)
    assert wrapper.fit(train, labels, seed=9, fit_threads=2) is wrapper
    assert isinstance(wrapper.model, cls)
    assert wrapper.diagnostics["fit_threads"] == 2
    assert wrapper.diagnostics["iterations"] == 100

    def no_refit(*args, **kwargs):
        raise AssertionError("predict must reuse the existing model")

    monkeypatch.setattr(wrapper.model, "fit", no_refit)
    for scoring in [test, test[:7]]:
        predictions = wrapper.predict(scoring, prediction=prediction)
        truth = (
            expected.predict_proba(scoring)
            if prediction == "probability"
            else expected.predict(scoring, output_margin=prediction == "margin")
        )
        np.testing.assert_array_equal(predictions, truth)


def test_fit_predict_uses_public_fit_and_returns_independent_diagnostics(xgb, monkeypatch):
    train = np.arange(80.0).reshape(-1, 1)
    labels = (train[:, 0] >= 40).astype(int)
    wrapper = XGBoostModel()
    calls = []
    original = wrapper.fit

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(wrapper, "fit", record)
    scores, diagnostics = wrapper.fit_predict(train, labels, train[:10], seed=3)
    assert calls == [{"seed": 3, "fit_threads": 1}]
    np.testing.assert_array_equal(scores, wrapper.predict(train[:10]))
    assert diagnostics == wrapper.diagnostics
    diagnostics["fold"] = 7
    assert "fold" not in wrapper.diagnostics


def test_prediction_and_configuration_errors_are_not_statistical_failures():
    model = XGBoostModel()
    with pytest.raises(NotFittedError, match="fit"):
        model.predict(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="prediction"):
        model.predict(np.zeros((3, 1)), prediction="unknown")
    with pytest.raises(ValueError, match="classification"):
        XGBoostModel(task="regression").predict(np.zeros((3, 1)), prediction="probability")
    with pytest.raises(ValueError, match="task"):
        XGBoostModel(task="unknown")
    with pytest.raises(ValueError, match="device"):
        XGBoostModel(device="gpu")
    with pytest.raises(ValueError, match="fit_threads"):
        model.fit(np.zeros((3, 1)), np.zeros(3), seed=0, fit_threads=0)


def test_failed_refit_discards_previous_model_and_retains_engine_reason(xgb, monkeypatch):
    train = np.arange(20.0).reshape(-1, 1)
    model = XGBoostModel(task="regression").fit(train, train[:, 0], seed=0)

    def fail(*args, **kwargs):
        raise xgb.core.XGBoostError("refit failure")

    monkeypatch.setattr(xgb.XGBRegressor, "fit", fail)
    with pytest.raises(FitUnavailable) as failure:
        model.fit(train, train[:, 0], seed=1)
    assert failure.value.reason == "xgboost_fit_failed"
    assert failure.value.diagnostics["error"] == "refit failure"
    assert model.model is None and model.diagnostics == {}
    with pytest.raises(NotFittedError):
        model.predict(train)


def test_prediction_engine_failure_uses_shared_exception(xgb, monkeypatch):
    train = np.arange(20.0).reshape(-1, 1)
    model = XGBoostModel(task="regression").fit(train, train[:, 0], seed=0)

    def fail(*args, **kwargs):
        raise xgb.core.XGBoostError("prediction failure")

    monkeypatch.setattr(model.model, "predict", fail)
    with pytest.raises(FitUnavailable) as failure:
        model.predict(train)
    assert failure.value.diagnostics["error"] == "prediction failure"
    assert failure.value.diagnostics["xgboost_version"] == xgb.__version__


def test_cuda_requires_gpu_build_before_constructing_a_model(xgb, monkeypatch):
    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": False})
    with pytest.raises(ImportError, match="CUDA-enabled"):
        XGBoostModel(device="cuda:1").fit(np.zeros((4, 1)), [0, 1, 0, 1], seed=0)


def test_requested_cuda_cannot_silently_fall_back_to_cpu(xgb, monkeypatch):
    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": True})
    original = xgb.XGBClassifier
    monkeypatch.setattr(
        xgb, "XGBClassifier", lambda **options: original(**dict(options, device="cpu"))
    )
    with pytest.raises(RuntimeError, match="selected 'cpu'"):
        XGBoostModel(device="cuda:0").fit(np.arange(20.0).reshape(-1, 1), [0, 1] * 10, seed=0)


def test_unindexed_cuda_follows_current_torch_device(monkeypatch):
    from types import SimpleNamespace

    from tabgen_eval.execution.torch import tensors

    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 2)
    monkeypatch.setattr(tensors, "torch_module", lambda **kwargs: SimpleNamespace(cuda=cuda))
    assert XGBoostModel.execution_device("cuda") == "cuda:2"
    assert XGBoostModel.execution_device("cuda:1") == "cuda:1"


@pytest.mark.parametrize("device", ["auto", "gpu", "cuda:-1", None])
def test_model_requires_explicit_device(device):
    with pytest.raises(ValueError, match="device must be"):
        XGBoostModel(device=device)
    with pytest.raises(ValueError, match="device must be"):
        XGBoostModel.execution_device(device)


def test_default_cpu_and_explicit_ordinal_do_not_probe_cuda(monkeypatch):
    from tabgen_eval.execution.torch import tensors

    def forbidden(**kwargs):
        pytest.fail("explicit CPU and CUDA ordinals must not initialize Torch")

    monkeypatch.setattr(tensors, "torch_module", forbidden)
    assert XGBoostModel().device == "cpu"
    assert XGBoostModel.execution_device("cpu") == "cpu"
    assert XGBoostModel.execution_device("cuda:2") == "cuda:2"


def _native_categorical_model(device="cpu"):
    """One numeric and one four-level categorical column; level 3 means missing."""
    rng = np.random.default_rng(481)
    categories = np.tile(np.arange(4, dtype=np.float32), 80)
    values = np.column_stack((rng.normal(size=len(categories)), categories)).astype(np.float32)
    labels = ((categories == 1) | ((categories == 3) & (values[:, 0] > 0))).astype(int)
    return XGBoostModel(device=device).fit(
        values,
        labels,
        seed=11,
        fit_threads=2,
        feature_types=["q", "c"],
        feature_names=["rate<raw>", "hotel"],
        categorical_cardinalities=[4],
    )


def test_native_categories_preserve_unseen_rest_semantics_and_singleton_splits(xgb):
    import json

    model = _native_categorical_model()
    test = np.array([[0, 0], [0, 1], [0, 2], [0, 4], [0, 5]], dtype=np.float32)
    scores = model.predict(test)
    # Unseen values match no learned level. Their path need not coincide with
    # an observed negative category's path across the whole forest.
    assert scores[3] == scores[4]
    assert scores[1] > scores[0]

    def check(node):
        if "leaf" in node:
            return
        if node["split"] == "f1":
            assert len(node["split_condition"]) == 1
        for child in node["children"]:
            check(child)

    booster = model.model.get_booster()
    for tree in booster.get_dump(dump_format="json"):
        check(json.loads(tree))
    assert model.model.max_cat_to_onehot == 5
    assert model.diagnostics["categorical_policy"] == "one-category-versus-rest"


def test_native_category_shap_and_serialization_retain_original_feature_mapping(xgb, tmp_path):
    import json

    model = _native_categorical_model()
    test = np.array([[0, 0], [0, 1], [1, 3], [-1, 3], [1, 4]], dtype=np.float32)
    matrix = xgb.DMatrix(test, feature_types=["q", "c"], enable_categorical=True)
    booster = model.model.get_booster()
    contributions = booster.predict(matrix, pred_contribs=True)
    interactions = booster.predict(matrix, pred_interactions=True)
    assert contributions.shape == (len(test), 3)
    assert interactions.shape == (len(test), 3, 3)
    np.testing.assert_allclose(contributions.sum(axis=1), model.predict(test), atol=3e-6)
    np.testing.assert_allclose(interactions.sum(axis=2), contributions, atol=3e-6)
    model_path = tmp_path / "native-categories.ubj"
    booster.save_model(model_path)
    restored = xgb.Booster(model_file=model_path)
    np.testing.assert_array_equal(restored.predict(matrix, output_margin=True), model.predict(test))
    metadata = json.loads(restored.attr("tabgen_eval_preprocessing"))
    assert metadata["feature_names"] == ["rate<raw>", "hotel"]
    assert metadata["categorical_cardinalities"] == [4]


@pytest.mark.parametrize(
    "metadata",
    [
        {"feature_types": ["q"]},
        {"feature_types": ["q", "wrong"]},
        {"feature_types": ["q", "c"], "categorical_cardinalities": []},
        {"feature_types": ["q", "c"], "categorical_cardinalities": [0]},
        {"feature_types": ["q", "c"], "categorical_cardinalities": [2**24]},
        {"feature_names": ["a", "b"]},
    ],
)
def test_native_category_metadata_errors_are_not_statistical_failures(metadata):
    with pytest.raises(ValueError):
        XGBoostModel().fit(np.zeros((4, 2)), [0, 1, 0, 1], seed=0, **metadata)
