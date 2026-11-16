"""Automatic XGBoost placement checks dependencies and memory before transfers."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, evaluate
from tabgen_eval.execution.xgboost import device as routing
from tabgen_eval.metrics import MLE, NonLinearC2ST
from tabgen_eval.models import XGBoostModel

pytestmark = pytest.mark.unit


@pytest.fixture
def gpu_environment(monkeypatch):
    from tabgen_eval.execution.torch import context, tensors

    calls = []

    @contextmanager
    def current_device(index):
        calls.append(("cupy_device", index))
        yield

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 2),
        empty=lambda n, device: calls.append(("torch_buffer", device)) or object(),
    )
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(Device=current_device),
        from_dlpack=lambda tensor: calls.append(("bridge", tensor)) or object(),
    )
    engine = SimpleNamespace(build_info=lambda: {"USE_CUDA": True})
    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: engine))
    monkeypatch.setattr(tensors, "torch_module", lambda **kwargs: torch)
    monkeypatch.setattr(context, "triton_installed", lambda: True)
    monkeypatch.setattr(context, "cuda_supported", lambda module: True)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    return engine, torch, cupy, calls


def test_gpu_probe_uses_current_device_and_zero_copy_bridge(gpu_environment):
    _, _, _, calls = gpu_environment
    assert routing.cuda_device() == ("cuda:2", "xgboost_cuda_available")
    assert calls[0] == ("cupy_device", 2)
    assert calls[1] == ("torch_buffer", "cuda:2")
    assert calls[2][0] == "bridge"


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("cpu_build", "xgboost_cuda_unavailable"),
        ("no_gpu", "cuda_unavailable"),
        ("missing_cupy", "cupy_unavailable"),
        ("old_cupy", "cupy_bridge_unavailable"),
        ("broken_bridge", "cuda_environment_unavailable"),
        ("missing_triton", "triton_unavailable"),
        ("unsupported_stack", "cuda_version_below_minimum"),
    ],
)
def test_gpu_dependency_failures_are_cpu_fallbacks(gpu_environment, monkeypatch, problem, reason):
    engine, torch, cupy, _ = gpu_environment
    if problem == "cpu_build":
        engine.build_info = lambda: {"USE_CUDA": False}
    elif problem == "no_gpu":
        torch.cuda.is_available = lambda: False
    elif problem == "missing_cupy":
        monkeypatch.setitem(sys.modules, "cupy", None)
    elif problem == "old_cupy":
        cupy.from_dlpack = None
    elif problem in {"missing_triton", "unsupported_stack"}:
        from tabgen_eval.execution.torch import context

        attribute = "triton_installed" if problem == "missing_triton" else "cuda_supported"
        monkeypatch.setattr(context, attribute, lambda *args: False)
    else:

        def broken(tensor):
            raise RuntimeError("incompatible CUDA runtime")

        cupy.from_dlpack = broken
    assert routing.cuda_device() == (None, reason)


@pytest.mark.parametrize("metric_type", [NonLinearC2ST, MLE])
@pytest.mark.parametrize("problem", [None, "too_small", "memory_error"])
def test_automatic_selection_checks_memory_and_records_reason(monkeypatch, metric_type, problem):
    metric = metric_type()
    calls = []
    monkeypatch.setattr(routing, "cuda_device", lambda: ("cuda:2", "xgboost_cuda_available"))

    def plan(metric, device, **kwargs):
        calls.append(device)
        if device.startswith("cuda") and problem == "memory_error":
            raise MemoryError("insufficient memory")
        budget = 1 if device.startswith("cuda") and problem == "too_small" else 8192
        metric.gpu_workspace_mb = metric.memory_limit_mb = budget
        return SimpleNamespace(task_workspace_bytes=budget * 1024**2)

    monkeypatch.setattr(routing, "plan_metric_memory", plan)
    selected = routing.select_device(metric, rows=100, numeric=2, cardinalities=(4,))
    assert selected == ("cuda:2" if problem is None else "cpu")
    assert routing.resolved_device(metric) == selected
    assert calls == (["cuda:2"] if problem is None else ["cuda:2", "cpu"])
    assert metric.device == "auto"
    assert (
        metric._xgboost_routing_reason
        == {
            None: "xgboost_cuda_available",
            "too_small": "xgboost_workspace_too_small",
            "memory_error": "insufficient_free_cuda_memory",
        }[problem]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_explicit_device_never_uses_automatic_fallback(monkeypatch, device):
    metric = MLE(device=device)

    def forbidden():
        pytest.fail("explicit device must not run automatic dependency probes")

    monkeypatch.setattr(routing, "cuda_device", forbidden)
    monkeypatch.setattr(routing, "plan_metric_memory", lambda *args, **kwargs: None)
    assert routing.select_device(metric, rows=100, numeric=2, cardinalities=()) == device
    assert metric._xgboost_routing_reason == "explicit_device"


@pytest.mark.parametrize("selected_device", ["cpu", "cuda:2"])
def test_custom_nonlinear_fallback_passes_resolved_device_to_models(monkeypatch, selected_device):
    from tabgen_eval.execution.runtime.dispatch import supports

    class CustomNonLinearC2ST(NonLinearC2ST):
        pass

    metric = CustomNonLinearC2ST()
    assert not supports(metric)
    selections, fits = [], []

    def select(current, **kwargs):
        assert current is metric and current.device == "auto"
        selections.append(current)
        current._xgboost_device = selected_device
        current._xgboost_routing_reason = "test_selection"
        return selected_device

    def fit_predict(model, train, labels, test, **kwargs):
        assert model.device == selected_device
        fits.append(model.device)
        return np.zeros(len(test)), {"converged": True, "device": model.device}

    monkeypatch.setattr(routing, "select_device", select)
    monkeypatch.setattr(XGBoostModel, "fit_predict", fit_predict)
    reference = pd.DataFrame({"x": np.arange(18, dtype=float)})
    candidate = pd.DataFrame({"x": np.arange(18, dtype=float) + 0.25})
    folds = np.arange(len(reference)) % metric.cv_folds
    outcome = metric.compute_partitioned(
        reference,
        candidate,
        Schema(numeric=["x"]),
        folds,
        folds,
        seed=41,
    )
    assert selections and fits == [selected_device] * metric.cv_folds
    assert outcome.value == 0.5
    assert all(info["device"] == selected_device for info in outcome.folds)
    assert metric.device == "auto"


@pytest.mark.parametrize("metric_type", [NonLinearC2ST, MLE])
@pytest.mark.parametrize(
    "requested,reported",
    [("auto", "unresolved"), ("cuda", "unresolved"), ("cpu", "cpu"), ("cuda:2", "cuda:2")],
)
def test_unexecuted_metadata_does_not_probe_or_assume_cpu(
    monkeypatch, metric_type, requested, reported
):
    from tabgen_eval.execution.runtime.metadata import execution_info
    from tabgen_eval.execution.torch import tensors

    def forbidden(*args, **kwargs):
        pytest.fail("reporting an unexecuted metric must not resolve or probe its device")

    monkeypatch.setattr(routing, "cuda_device", forbidden)
    monkeypatch.setattr(routing, "select_device", forbidden)
    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(forbidden))
    monkeypatch.setattr(tensors, "torch_module", forbidden)
    metric = metric_type(device=requested)
    assert routing.resolved_device(metric) == reported
    assert execution_info(metric) == {"implementation": "xgboost", "device": reported}
    assert not hasattr(metric, "_xgboost_device")


@pytest.mark.parametrize(
    "metric,test_only", [("nonlinear_c2st", False), ("mle", False), ("mle", True)]
)
def test_auto_without_cupy_runs_actual_cpu_fits(monkeypatch, metric, test_only):
    xgb = pytest.importorskip("xgboost", minversion="2.1")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", None)
    rng = np.random.default_rng(892)
    real = pd.DataFrame({"x": rng.normal(size=80)})
    real["y"] = real.x * 2
    generated = real.sample(frac=1, random_state=4).reset_index(drop=True)
    data = EvalData(
        test=real,
        generated=None if test_only else generated,
        schema=Schema(numeric=["x", "y"], target="y"),
    )
    original = XGBoostModel.fit
    fitted = []

    def fit(self, train, labels, **kwargs):
        assert self.device == "cpu" and train.device.type == "cpu"
        fitted.append(self.device)
        return original(self, train, labels, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    report = evaluate(
        data, metrics=[metric], uncertainty={"mode": "reference", "n_replicates": 2}, n_jobs=1
    )
    info = report.provenance["metric_execution"][metric]
    assert fitted and info["device"] == "cpu"
    assert info["routing_reason"] == "cupy_unavailable"
    assert report.options_dict["device"] == "auto"
    if test_only:
        assert np.isfinite(report[metric].reference_distribution.samples).all()
    else:
        assert np.isfinite(report[metric].value)
