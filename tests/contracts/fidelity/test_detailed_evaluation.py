"""Point detail capture preserves numerical values, sampling and computation count."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.preprocessing.categories import category_key
from tabgen_eval.preprocessing.columns import PreparedColumns
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options


def _frames():
    rng = np.random.default_rng(391)
    real = pd.DataFrame({"a": rng.normal(size=420), "b": rng.normal(size=420)})
    generated = pd.DataFrame({"a": rng.normal(0.2, size=240), "b": rng.normal(size=240)})
    labels = np.empty(4, dtype=object)
    labels[:] = [1, 1.0, "1", " A "]
    real["label"] = pd.Series(labels[np.arange(len(real)) % 4], dtype=object)
    generated["label"] = pd.Series(labels[np.arange(len(generated)) % 4], dtype=object)
    return real, generated, Schema(numeric=["a", "b"], categorical=["label"])


@pytest.mark.parametrize("n_bootstrap", [0, 2])
def test_python_capture_preserves_full_scalar_report(n_bootstrap):
    real, generated, schema = _frames()
    cfg = make_options(
        uncertainty="both" if n_bootstrap else "none",
        n_replicates=max(n_bootstrap, 1),
        seed=56,
    )
    metrics = ["shape", "trend", "conditional_fidelity"]
    plain = evaluate(real, generated, schema, metrics=metrics, options=cfg)
    detailed = evaluate(real, generated, schema, metrics=metrics, options=cfg, details=True)
    pd.testing.assert_frame_equal(plain.to_dataframe(), detailed.to_dataframe(), check_exact=True)
    assert "details" not in json.loads(plain.to_json())
    assert "details" in json.loads(detailed.to_json())
    assert len(detailed.details("shape")) == 3
    assert len(detailed.details("trend")) == 3
    for result in detailed.results:
        assert result.details.metadata["scope"] == (
            "generated_vs_reference_replicate" if n_bootstrap else "generated_vs_reference_point"
        )
        assert result.details.metadata["n_reference"] == 210
        assert result.details.metadata["n_candidate"] == 210


def test_capture_does_not_repeat_shape_primitives(monkeypatch):
    from tabgen_eval.execution.torch.cpu.executor import CppExecutor

    real, generated, schema = _frames()
    calls = []

    def instrument(name):
        original = getattr(CppExecutor, name)

        def counted(self, *args, **kwargs):
            calls.append(1 if name == "compute_detailed" else len(args[1]))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(CppExecutor, name, counted)

    instrument("compute")
    instrument("compute_detailed")
    cfg = make_options(uncertainty="both", n_replicates=2)
    evaluate(real, generated, schema, metrics=["shape"], options=cfg)
    expected = sum(calls)
    assert expected > 0
    calls.clear()
    evaluate(real, generated, schema, metrics=["shape"], options=cfg, details=True)
    assert sum(calls) == expected == 4


def test_details_first_repetition_is_independent_of_repetition_count():
    real, generated, schema = _frames()
    reports = [
        evaluate(
            real,
            generated,
            schema,
            metrics=["shape", "trend", "conditional_fidelity"],
            options=make_options(uncertainty="both" if b else "none", n_replicates=max(b, 1)),
            details=True,
        )
        for b in (2, 4)
    ]
    for metric, level in (("shape", None), ("trend", None), ("conditional_fidelity", "slice")):
        pd.testing.assert_frame_equal(
            reports[0].details(metric, level=level),
            reports[1].details(metric, level=level),
            check_exact=True,
        )


def test_tensor_report_preserves_scalar_results_and_typed_slice_labels(dtype, device):
    real, generated, schema = _frames()
    cfg = make_options(uncertainty="both", n_replicates=2, device=device, dtype=dtype)
    metrics = ["shape", "trend", "conditional_fidelity"]
    plain = evaluate(real, generated, schema, metrics=metrics, options=cfg)
    detailed = evaluate(real, generated, schema, metrics=metrics, options=cfg, details=True)
    pd.testing.assert_frame_equal(plain.to_dataframe(), detailed.to_dataframe(), check_exact=True)
    slices = detailed.details("conditional_fidelity", level="slice")
    labels = slices.loc[slices["column"] == "label", "label"]
    assert {category_key(value) for value in labels} == {
        category_key(1),
        category_key(1.0),
        category_key("1"),
        category_key(" A "),
    }


@pytest.mark.parametrize("arrow", [False, True])
def test_preparation_retains_existing_typed_dictionary_only_when_requested(arrow):
    real, generated, schema = _frames()
    if arrow:
        pytest.importorskip("pyarrow")
        from tabgen_eval.preprocessing.arrow import try_prepare

        plain = try_prepare(real, generated, schema)
        detailed = try_prepare(real, generated, schema, retain_labels=True)
    else:
        plain = PreparedColumns.prepare(real, generated, schema)
        detailed = PreparedColumns.prepare(real, generated, schema, retain_labels=True)
    assert plain.category_values is None
    assert [category_key(v) for v in detailed.category_values[0]] == [
        category_key(1),
        category_key(1.0),
        category_key("1"),
        category_key(" A "),
    ]
    np.testing.assert_array_equal(plain.reference.categorical, detailed.reference.categorical)
    np.testing.assert_array_equal(plain.candidate.categorical, detailed.candidate.categorical)


@pytest.mark.parametrize("value", [1, "yes", None, []])
def test_details_requires_boolean(value):
    real, generated, schema = _frames()
    with pytest.raises(ValueError, match="details must be a boolean"):
        evaluate(real, generated, schema, details=value)
