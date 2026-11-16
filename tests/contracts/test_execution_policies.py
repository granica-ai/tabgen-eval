"""Training-referenced DCR and configurable model execution policies."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import MLE, LinearC2ST, NonLinearC2ST
from tabgen_eval.models import XGBoostModel
from tabgen_eval.registry.registry import resolve_metrics
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("metric", [NonLinearC2ST, MLE])
@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_model_execution_accepts_direct_and_global_device_options(metric, device):
    assert metric(device=device).device == device
    resolved = resolve_metrics([metric.name], make_options(device=device))[0]
    assert resolved.device == device
    with pytest.raises(TypeError, match="unexpected kwarg"):
        metric(fixed_execution_options={})


def test_global_gpu_options_still_apply_to_other_metrics():
    metrics = resolve_metrics(
        ["nonlinear_c2st", "mle", "linear_c2st"],
        make_options(device="cuda:1", dtype="float32"),
    )
    assert [(m.device, m.dtype) for m in metrics] == [
        ("cuda:1", "float32"),
        ("cuda:1", "float32"),
        ("cuda:1", "float32"),
    ]
    assert not LinearC2ST.fixed_execution_options


@pytest.mark.parametrize("per_metric", [False, True])
@pytest.mark.parametrize("metric", ["nonlinear_c2st", "mle"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fp32_configuration_runs_actual_cpu_fits(metric, task, per_metric, monkeypatch):
    pytest.importorskip("xgboost", minversion="2.1")
    observed = []
    original = XGBoostModel.fit

    def fit(self, train, labels, **kwargs):
        assert self.device == "cpu"
        import torch

        assert train.dtype == torch.float32
        observed.append(len(train))
        return original(self, train, labels, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    rng = np.random.default_rng(63)
    frames = []
    for _ in range(3):
        x = rng.normal(size=120)
        frames.append(
            pd.DataFrame({"x": x, "y": (x > 0).astype(str) if task == "classification" else 2 * x})
        )
    schema = Schema(
        numeric=["x"] if task == "classification" else ["x", "y"],
        categorical=["y"] if task == "classification" else [],
        target="y",
    )
    cfg = make_options(
        device="cpu",
        dtype="float64" if per_metric else "float32",
        uncertainty="both",
        n_replicates=2,
    )
    metric_cls = MLE if metric == "mle" else NonLinearC2ST
    resolved = (
        configured_metric(metric_cls, cfg, dtype="float32")
        if per_metric
        else resolve_metrics([metric], cfg)[0]
    )
    assert resolved.device == "cpu" and resolved.dtype == "float32"
    if metric == "mle":
        report = evaluate(frames[0], frames[2], schema, metrics=[resolved], options=cfg)
        assert len(observed) == 6  # two point fits and two fits per bootstrap
    else:
        report = evaluate(frames[0], frames[2], schema, metrics=[resolved], options=cfg)
        assert len(observed) == 18  # two points and two paired repetitions, three folds each
    assert np.isfinite(report[f"{metric}/" + ("ratio" if metric == "mle" else "auc")].value)
    execution = report.provenance["metric_execution"][metric]
    assert execution["implementation"] == "xgboost"
    assert execution["device"] == "cpu"
    if metric == "nonlinear_c2st":
        assert execution["dtype"] == "float32"
        assert execution["engine_feature_precision"] == "float32"


def test_dcr_requires_training_role_and_uses_full_point_only_pools():
    training = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]})
    generated = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
    holdout = pd.DataFrame({"x": [8.0, 10.0]})
    schema = Schema(numeric=["x"])
    report = evaluate_memorization(
        training,
        generated,
        schema,
        holdout=holdout,
        metrics=["dcr"],
        options=make_options(
            uncertainty="both",
            n_replicates=99,
        ),
    )
    assert report["dcr/median"].value == pytest.approx(1 / 6)
    assert report["dcr/ratio"].value == pytest.approx(1 / 3)
    assert report.provenance["n_selected_training"] == 4
    assert report.provenance["n_selected_generated"] == 3
    assert report.provenance["n_selected_holdout"] == 2
    assert report.provenance["uncertainty"] == "none"
    for _, stat in report.iter_stats():
        assert stat.generated_distribution is None and stat.reference_distribution is None
        assert np.isnan(stat.ci_low) and np.isnan(stat.ci_high)
    with pytest.raises(ValueError, match="comparison metrics"):
        evaluate(holdout, generated, schema, metrics=["dcr"])
