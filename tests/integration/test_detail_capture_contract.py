"""Capture boundaries, unavailable points, and optional backend isolation."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.details import DetailedOutcome, DetailTable, FidelityDetails
from tabgen_eval.metrics import Conditional, Shape, Trend
from tabgen_eval.preprocessing.columns import PreparedColumns
from tabgen_eval.report import MetricResult, Report
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def _frames(n=240):
    rng = np.random.default_rng(82)
    real = pd.DataFrame(
        {"x": rng.normal(size=n), "y": rng.normal(size=n), "c": rng.choice(["a", "b"], n)}
    )
    generated = real.iloc[: n // 2].copy()
    return real, generated, Schema(numeric=["x", "y"], categorical=["c"])


def test_uncomputed_point_does_not_claim_detail_capture_or_recommend_only_enabling_it():
    real, generated, schema = _frames(30)
    options = make_options()
    chosen = [Shape(n_min=100)]
    ordinary = evaluate(real, generated, schema, metrics=chosen, options=options)
    captured = evaluate(real, generated, schema, metrics=chosen, options=options, details=True)
    pd.testing.assert_frame_equal(
        ordinary.to_dataframe(), captured.to_dataframe(), check_exact=True
    )
    assert "INSUFFICIENT_DATA" in captured["shape"].flag_codes
    assert captured.results[0].details is None
    with pytest.raises(ValueError, match="computed point"):
        captured.details("shape")


def test_nonfidelity_metrics_are_unchanged_when_capture_is_enabled():
    real, generated, schema = _frames()
    options = make_options(uncertainty="none")
    ordinary = evaluate(real, generated, schema, metrics=["prdc"], options=options)
    captured = evaluate(real, generated, schema, metrics=["prdc"], options=options, details=True)
    pd.testing.assert_frame_equal(
        ordinary.to_dataframe(), captured.to_dataframe(), check_exact=True
    )
    assert captured.results[0].details is None


def test_capture_preserves_caller_frames_and_exact_categorical_content():
    real, generated, schema = _frames()
    real.loc[real.index[::3], "c"] = " A "
    generated.loc[generated.index[::3], "c"] = " A "
    before_real, before_generated = real.copy(deep=True), generated.copy(deep=True)
    evaluate(
        real,
        generated,
        schema,
        metrics=["shape", "trend", "conditional_fidelity"],
        options=make_options(uncertainty="none"),
        details=True,
    )
    pd.testing.assert_frame_equal(real, before_real, check_exact=True)
    pd.testing.assert_frame_equal(generated, before_generated, check_exact=True)


def test_cpu_details_public_path_does_not_load_optional_gpu_or_model_engines():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'triton', 'xgboost'}:
            raise AssertionError('unexpected optional engine import: ' + fullname)
sys.meta_path.insert(0, Block())
import numpy as np
import pandas as pd
from tabgen_eval import Schema
from tests.support.options import make_options
from tests.support.evaluation import paired_report as evaluate
real = pd.DataFrame({'x': np.arange(240.), 'y': np.arange(240.) ** 2, 'c': [' A ', 'nan'] * 120})
report = evaluate(real, real.iloc[:120], Schema(numeric=['x', 'y'], categorical=['c']),
    metrics=['shape', 'trend', 'conditional_fidelity'],
    options=make_options(uncertainty="none"), details=True)
assert len(report.details('shape')) == 3
assert len(report.details('trend')) == 3
assert len(report.details('conditional_fidelity', level='slice')) > 0
assert 'details' in report.to_json()
assert 'torch' in sys.modules
assert not {'triton', 'xgboost'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, timeout=30)


def test_context_without_category_vocabularies_fails_before_native_dispatch(monkeypatch):
    pytest.importorskip("torch")
    from tabgen_eval.execution.torch.context import TorchContext

    real, generated, schema = _frames()
    context = TorchContext.from_columns(PreparedColumns.prepare(real, generated, schema))

    def forbidden(*args, **kwargs):
        raise AssertionError("native executor should not load without required labels")

    monkeypatch.setattr(context, "_executor", forbidden)
    with pytest.raises(ValueError, match="retain_labels=True"):
        context.compute_detailed(Conditional(device="cpu"), np.arange(120), np.arange(120))


@pytest.mark.parametrize("metric", [Shape(), Trend()])
def test_shape_and_trend_do_not_require_retained_category_vocabularies(monkeypatch, metric):
    pytest.importorskip("torch")
    from tabgen_eval.execution.torch.context import TorchContext

    real, generated, schema = _frames()
    context = TorchContext.from_columns(PreparedColumns.prepare(real, generated, schema))
    calls = []

    def compute(*args, **kwargs):
        calls.append(1)
        return DetailedOutcome({metric.stat_names[0]: 0.5}, FidelityDetails())

    monkeypatch.setattr(context, "_executor", lambda _: SimpleNamespace(compute_detailed=compute))
    outcome = context.compute_detailed(metric, np.arange(120), np.arange(120))
    assert calls == [1]
    assert outcome.details.metadata["column_names"] == ("x", "y", "c")


def test_raw_slice_tables_without_label_metadata_have_an_explicit_rendering_error():
    details = FidelityDetails(
        {
            "slices": DetailTable(
                {
                    "column_index": np.array([0]),
                    "label_code": np.array([0]),
                    "lower": np.array([np.nan]),
                    "upper": np.array([np.nan]),
                    "left_closed": np.array([False]),
                    "right_closed": np.array([False]),
                }
            )
        },
        {"column_names": ("category",)},
    )
    report = Report(
        [MetricResult("conditional_fidelity", "conditional_fidelity", details=details)], {}, {}
    )
    with pytest.raises(ValueError, match="retained category vocabulary"):
        report.details("conditional_fidelity", level="slice")
