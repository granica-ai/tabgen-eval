"""Small native CUDA classifier/utility reports, including serial bootstrap fits."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize(
    "metric,test_only", [("nonlinear_c2st", False), ("mle", False), ("mle", True)]
)
def test_auto_uses_gpu_for_points_and_repetitions(torch, monkeypatch, metric, test_only):
    from tabgen_eval import EvalData
    from tabgen_eval import evaluate as evaluate_public
    from tabgen_eval.models import XGBoostModel

    xgb = pytest.importorskip("xgboost", minversion="2.1")
    pytest.importorskip("cupy")
    if not xgb.build_info().get("USE_CUDA"):
        pytest.skip("CUDA-enabled XGBoost required")
    rng = np.random.default_rng(987)
    real = pd.DataFrame({"x": rng.normal(size=80), "c": rng.choice(["a", "b"], 80)})
    real["y"] = real.x * 2
    generated = real.sample(frac=1, random_state=4).reset_index(drop=True)
    data = EvalData(
        test=real,
        generated=None if test_only else generated,
        schema=Schema(numeric=["x", "y"], categorical=["c"], target="y"),
    )
    expected = f"cuda:{torch.cuda.current_device()}"
    original = XGBoostModel.fit
    fits = []
    parent = os.getpid()

    def fit(self, train, labels, **kwargs):
        assert os.getpid() == parent
        assert self.device == expected and str(train.device) == expected
        result = original(self, train, labels, **kwargs)
        assert result.diagnostics["device"] == expected
        fits.append(expected)
        return result

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    report = evaluate_public(
        data,
        metrics=[metric],
        uncertainty={"mode": "reference" if test_only else "candidate", "n_replicates": 2},
        n_jobs=4,
    )
    execution = report.provenance["metric_execution"][metric]
    assert fits and execution["device"] == expected
    assert execution["routing_reason"] == "xgboost_cuda_available"
    assert report.options_dict["device"] == "auto"
    scores = report[metric]
    distribution = scores.reference_distribution if test_only else scores.generated_distribution
    assert len(distribution.samples) == 2 and np.isfinite(distribution.samples).all()


@pytest.mark.parametrize(
    ("metric", "task"),
    [("nonlinear_c2st", "classification"), ("mle", "classification"), ("mle", "regression")],
)
def test_native_cuda_reports_follow_device_and_keep_bootstraps_serial(metric, task, monkeypatch):
    xgb = pytest.importorskip("xgboost", minversion="2.1")
    torch = pytest.importorskip("torch")
    if not xgb.build_info().get("USE_CUDA") or not torch.cuda.is_available():
        pytest.skip("CUDA-enabled XGBoost and an available GPU are required")
    device = f"cuda:{torch.cuda.current_device()}"

    def no_process_workers(*args, **kwargs):
        pytest.fail("GPU bootstrap fits must not spawn competing process workers")

    if metric == "mle":
        from tabgen_eval.execution.runtime import utility

        original_fit = utility.fit_predictor
        parent_pid = os.getpid()
        fit_calls = []

        def serial_fit(*args, **kwargs):
            assert os.getpid() == parent_pid, "GPU MLE fits must run in the caller process"
            fit_calls.append(1)
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(utility, "fit_predictor", serial_fit)
    else:
        from tabgen_eval.execution.runtime import comparisons

        monkeypatch.setattr(comparisons, "Parallel", no_process_workers)
    rng = np.random.default_rng(1631)
    r, s = pd.DataFrame({"x": rng.normal(size=240)}), pd.DataFrame({"x": rng.normal(size=240)})
    if metric == "mle":
        for frame in (r, s):
            frame["target"] = (frame.x > 0).astype(str) if task == "classification" else 2 * frame.x
        schema = Schema(
            numeric=["x"] if task == "classification" else ["x", "target"],
            categorical=["target"] if task == "classification" else [],
            target="target",
        )
    else:
        schema = Schema(numeric=["x"])
    options = make_options(
        device=device,
        n_jobs=2,
        uncertainty="both",
        n_replicates=2,
    )
    if metric == "mle":
        report = evaluate(r, s, schema, metrics=[metric], options=options)
        execution = report.provenance["metric_execution"][metric]
        assert execution["implementation"] == "xgboost" and execution["device"] == device
        assert np.isfinite(report["mle/ratio"].value)
        score, ratio = report["mle/score"], report["mle/ratio"]
        expected_direction = "lower_better" if task == "regression" else "higher_better"
        assert score.direction.value == ratio.direction.value == expected_direction
        assert ratio.value == pytest.approx(score.value / score.ideal)
        assert report["mle/ratio"].generated_distribution.finite_count == 2
        info = report.results[0].diagnostics
        fits = [
            fit
            for outcome in [info["point"], *info["replicates"]]
            for fit in outcome["diagnostics"]["fits"].values()
        ]
        assert all(fit["device"] == device for fit in fits)
        assert len(fit_calls) == 6
        return
    report = evaluate(r, s, schema, metrics=[metric], options=options)
    execution = report.provenance["metric_execution"][metric]
    assert execution["implementation"] == "xgboost" and execution["device"] == device
    assert execution["implementation"] == "xgboost"
    assert execution["engine_feature_precision"] == "float32"
    assert all(np.isfinite(stat.value) for stat in report.results[0].stats.values())
    info = report.provenance["classifier_diagnostics"][metric]
    fits = [fit for comparison in info["point"].values() for fit in comparison["folds"]]
    fits += [
        fit
        for replicas in info["replicates"].values()
        for replica in replicas
        for fit in replica["folds"]
    ]
    assert fits and all(fit["device"] == device for fit in fits)


def test_native_categories_use_zero_copy_device_inputs_and_original_column_shap(tmp_path):
    """Exercise producer streams, unseen values, explicit missing, and reload."""
    import json

    from tabgen_eval.execution.xgboost.arrays import engine_array
    from tabgen_eval.models import XGBoostModel

    xgb = pytest.importorskip("xgboost", minversion="2.1")
    torch = pytest.importorskip("torch")
    cupy = pytest.importorskip("cupy")
    if not xgb.build_info().get("USE_CUDA") or not torch.cuda.is_available():
        pytest.skip("CUDA-enabled XGBoost and an available GPU are required")
    device = f"cuda:{torch.cuda.current_device()}"
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        categories = torch.arange(4, device=device).repeat(80)
        # Level 3 explicitly represents missing. It is never passed as NaN.
        train = torch.column_stack((torch.zeros_like(categories), categories)).float()
        labels = ((categories == 1) | (categories == 3)).float()
        view = engine_array(train)
        assert view.data.ptr == train.data_ptr()
        assert view.device.id == train.device.index
        with cupy.cuda.Device(train.device.index):
            # DLPack provides the stream dependency; no global synchronize here.
            model = XGBoostModel(device=device).fit(
                train,
                labels,
                seed=1,
                fit_threads=2,
                feature_types=["q", "c"],
                feature_names=["rate", "hotel"],
                categorical_cardinalities=[4],
            )
        test = torch.tensor(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], dtype=torch.float32, device=device
        )
        scores = model.predict(test)
        assert min(scores[1], scores[3]) > max(scores[0], scores[2])
        # An unseen value follows "rest" at every equality split. That path
        # need not coincide with any particular observed category's full path.
        leaves = []
        for text in model.model.get_booster().get_dump(dump_format="json"):
            node = json.loads(text)
            while "leaf" not in node:
                assert node["split"] == "f1"
                assert len(node["split_condition"]) == 1
                assert 4 not in node["split_condition"]
                node = next(child for child in node["children"] if child["nodeid"] == node["no"])
            leaves.append(node["leaf"])
        # Balanced labels give initial log-odds zero.
        np.testing.assert_allclose(sum(leaves), scores[4], atol=3e-6)
        matrix = xgb.DMatrix(engine_array(test), feature_types=["q", "c"], enable_categorical=True)
        booster = model.model.get_booster()
        contributions = booster.predict(matrix, pred_contribs=True)
        interactions = booster.predict(matrix, pred_interactions=True)
        assert contributions.shape == (len(test), 3)
        assert interactions.shape == (len(test), 3, 3)
        # Native FP32 attribution and prediction sum tree contributions in
        # different orders; accumulated roundoff across 100 trees is expected.
        np.testing.assert_allclose(contributions.sum(axis=1), scores, atol=2e-5)
        np.testing.assert_allclose(interactions.sum(axis=2), contributions, atol=2e-5)
        path = tmp_path / "native-gpu.ubj"
        booster.save_model(path)
        restored = xgb.Booster(model_file=path)
        restored.set_param({"device": device})
        np.testing.assert_array_equal(restored.predict(matrix, output_margin=True), scores)
        assert json.loads(restored.attr("tabgen_eval_preprocessing"))["feature_names"] == [
            "rate",
            "hotel",
        ]
