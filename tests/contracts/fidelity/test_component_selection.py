"""Public selectors retain shared-job values and skip unrelated calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.components import CONDITIONAL_COMPONENTS, PRDC_COMPONENTS
from tabgen_eval.execution.runtime.dispatch import compute_python_details
from tabgen_eval.metrics import PRDC, Conditional

pytestmark = pytest.mark.unit


def frames():
    rng = np.random.default_rng(431)
    tables = []
    for size in (73, 61):
        tables.append(
            pd.DataFrame(
                {
                    "x": rng.normal(size=size),
                    "y": rng.normal(size=size),
                    "c": rng.choice(["a", "b", "c"], size=size),
                }
            )
        )
    return *tables, Schema(numeric=["x", "y"], categorical=["c"])


def options(backend, dtype):
    return dict(
        device="auto" if backend == "python" else backend,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "metric_class,components", [(Conditional, CONDITIONAL_COMPONENTS), (PRDC, PRDC_COMPONENTS)]
)
def test_selected_components_preserve_full_job_values(
    backend, backend_dtype, metric_class, components
):
    reference, candidate, schema = frames()
    kwargs = options(backend, backend_dtype)
    if metric_class is Conditional:
        kwargs.update(min_count=1, rare_threshold=0.4)
    full = metric_class(**kwargs).compute(reference, candidate, schema, rng=None)
    for component in components:
        selected = metric_class(requested_components=(component,), **kwargs)
        actual = selected.compute(reference, candidate, schema, rng=None)
        assert tuple(actual) == selected.stat_names
        for name, value in actual.items():
            if name == component or name.startswith(component + "_"):
                assert value == pytest.approx(full[name], nan_ok=True)
            else:
                assert np.isnan(value)


@pytest.mark.parametrize("component", CONDITIONAL_COMPONENTS)
def test_python_conditional_does_not_compute_unselected_slices(monkeypatch, component):
    from tabgen_eval.execution.python.conditional import _Encoded

    def unexpected(*args, **kwargs):
        pytest.fail("unselected slice calculation ran")

    if component != "cond_shape":
        monkeypatch.setattr(_Encoded, "slice_shape", unexpected)
    if component != "cond_trend":
        monkeypatch.setattr(_Encoded, "slice_trend", unexpected)
    reference, candidate, schema = frames()
    metric = Conditional(min_count=1, requested_components=(component,))
    metric.compute(reference, candidate, schema, rng=None)
    compute_python_details(metric, reference, candidate, schema)


@pytest.mark.parametrize("component", PRDC_COMPONENTS)
def test_python_prdc_only_fits_required_radii(monkeypatch, component):
    from tabgen_eval.execution.python import distance

    original = distance.kth_neighbor_squared
    fitted = []

    def radii(features, k):
        fitted.append(features.n_rows)
        return original(features, k)

    monkeypatch.setattr(distance, "kth_neighbor_squared", radii)
    reference, candidate, schema = frames()
    metric = PRDC(requested_components=(component,))
    distance.prdc(reference, candidate, schema, metric.nearest_k, requested_components=(component,))
    assert fitted == [len(candidate) if component == "recall" else len(reference)]


def test_single_column_recovery_and_details(backend, backend_dtype):
    reference = pd.DataFrame({"c": ["a"] * 16 + ["b"] * 4 + [None]})
    candidate = pd.DataFrame({"c": ["a"] * 8 + ["b"] * 8 + ["new"] * 4 + [None]})
    schema = Schema(numeric=[], categorical=["c"])
    metric = Conditional(
        requested_components=("recovery",),
        min_count=1,
        rare_threshold=0.25,
        **options(backend, backend_dtype),
    )
    values = metric.compute(reference, candidate, schema, rng=None)
    assert values["recovery_all"] == pytest.approx(1.25)
    assert values["recovery_rare"] == pytest.approx(2.0)
    assert "recovery_all" not in metric.not_applicable(reference, candidate, schema)
    if backend == "python":
        detailed = compute_python_details(metric, reference, candidate, schema)
    else:
        from tabgen_eval.execution.torch.context import TorchContext
        from tabgen_eval.preprocessing.columns import PreparedColumns

        columns = PreparedColumns.prepare(reference, candidate, schema, retain_labels=True)
        context = TorchContext.from_columns(columns, device=backend, dtype=backend_dtype)
        assert "recovery_all" not in context.not_applicable(metric, schema)
        detailed = context.compute_detailed(
            metric, np.arange(len(reference)), np.arange(len(candidate))
        )
    assert detailed.values == pytest.approx(values, nan_ok=True)
    np.testing.assert_allclose(detailed.details.tables["slices"].columns["recovery"], [0.5, 2.0])


@pytest.mark.parametrize("component", PRDC_COMPONENTS)
def test_selected_prdc_sample_requirements_follow_radius_source(backend, backend_dtype, component):
    large = pd.DataFrame({"x": np.arange(7, dtype=float)})
    small = pd.DataFrame({"x": [0.5]})
    schema = Schema(numeric=["x"], categorical=[])
    reference, candidate = (small, large) if component == "recall" else (large, small)
    metric = PRDC(requested_components=(component,), **options(backend, backend_dtype))
    actual = metric.compute(reference, candidate, schema, rng=None)
    assert np.isfinite(actual[component])
    assert all(np.isnan(value) for name, value in actual.items() if name != component)


@pytest.mark.parametrize("metric_class", [Conditional, PRDC])
def test_unknown_components_raise(metric_class):
    with pytest.raises(ValueError, match="Unknown components"):
        metric_class(requested_components=("missing",))


@pytest.mark.gpu
@pytest.mark.parametrize("component", CONDITIONAL_COMPONENTS)
def test_cuda_conditional_skips_unselected_kernels(torch, monkeypatch, component):
    from tabgen_eval.execution.torch.fidelity import conditional
    from tests.support.backends import cuda_device

    def unexpected(*args, **kwargs):
        pytest.fail("unselected CUDA slice calculation ran")

    if component != "cond_shape":
        monkeypatch.setattr(conditional, "group_ks", unexpected)
    if component != "cond_trend":
        monkeypatch.setattr(conditional, "correlations", unexpected)
    if component == "recovery":
        monkeypatch.setattr(conditional, "group_gather", unexpected)
        monkeypatch.setattr(conditional, "pair_plan", unexpected)
    reference, candidate, schema = frames()
    metric = Conditional(
        device=cuda_device(torch),
        min_count=1,
        requested_components=(component,),
    )
    metric.compute(reference, candidate, schema, rng=None)


@pytest.mark.gpu
@pytest.mark.parametrize("component", PRDC_COMPONENTS)
def test_cuda_prdc_only_fits_required_radii(torch, monkeypatch, component):
    from tabgen_eval.execution.torch.distance import geometry
    from tests.support.backends import cuda_device

    original = geometry.kth_radii
    fitted = []

    def radii(features, k, workspace):
        fitted.append(features.n_rows)
        return original(features, k, workspace)

    monkeypatch.setattr(geometry, "kth_radii", radii)
    reference, candidate, schema = frames()
    metric = PRDC(device=cuda_device(torch), requested_components=(component,))
    metric.compute(reference, candidate, schema, rng=None)
    assert fitted == [len(candidate) if component == "recall" else len(reference)]
