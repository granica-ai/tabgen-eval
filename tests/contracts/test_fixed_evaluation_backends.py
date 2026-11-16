"""Fixed paired comparisons preserve backend arithmetic and shared preparation."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.dispatch import ExecutionSession
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tabgen_eval.metrics import PRDC, Conditional, LinearC2ST, Shape, Trend
from tabgen_eval.preprocessing.arrow import try_prepare
from tabgen_eval.schema import Schema
from tests.support.options import make_options, replace_options

pytestmark = pytest.mark.unit

METRICS = ["shape", "trend", "conditional_fidelity", "prdc", "linear_c2st"]
OPTIONS = {
    "conditional_fidelity": {"min_count": 5, "num_bins": 4, "rare_threshold": 0.2},
    "linear_c2st": {"solver": "direct"},
}


def _evaluate(*args, seed=None, options=None, **kwargs):
    from tests.support.evaluation import paired_report as evaluate

    options = options or make_options()
    options = replace_options(options, seed=options["seed"] if seed is None else seed)
    return evaluate(*args, options=options, **kwargs)


def _metric(report, name):
    return next(result for result in report.results if result.name == name)


def _values(result, *, reference=False):
    return {name: stat.ideal if reference else stat.value for name, stat in result.stats.items()}


def _frames():
    rng = np.random.default_rng(69113)
    frames = []
    for count, mean in [(321, 0.0), (147, 0.1)]:
        frame = pd.DataFrame(rng.normal(mean, 1, size=(count, 3)), columns=["x", "y", "z"])
        frame["y"] += 0.4 * frame["x"]
        frame["c"] = rng.choice(["A", " A ", "nan", "<MISSING>"], count)
        values = [1, 1.0, "1", True, None, "<MISSING>"]
        mixed = np.empty(count, dtype=object)
        mixed[:] = [values[i % len(values)] for i in range(count)]
        frame["typed"] = mixed
        frame.loc[::19, "x"] = np.nan
        frame.loc[::23, "z"] = np.nan
        frames.append(frame)
    return *frames, Schema(numeric=["x", "y", "z"], categorical=["c", "typed"])


def _config(device, dtype):
    return make_options(
        device=device,
        dtype=dtype,
        n_jobs=2,
    )


def _metric_instances(device, dtype):
    return [
        cls(device=device, dtype=dtype, **OPTIONS.get(cls.name, {}))
        for cls in [Shape, Trend, Conditional, PRDC, LinearC2ST]
    ]


def _classifier_protocol(metric, n, seed, comparison):
    metric_id = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(0xF17ED, metric_id, comparison))
    )
    rf, sf = source_folds(n, n, metric.cv_folds, rng)
    fit_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    return rf, sf, fit_seed


def _assert_backend_oracles(torch, device, dtype, monkeypatch):
    from tabgen_eval.execution.torch.tensors import TensorInputs

    real, generated, schema = _frames()
    originals = [real.copy(deep=True), generated.copy(deep=True)]
    options = _config(device, dtype)
    seed = 1197
    transfers = []
    from_columns = TensorInputs.from_columns.__func__

    def tracked(cls, columns, **kwargs):
        transfers.append((columns.reference.n_rows, columns.candidate.n_rows))
        assert not np.isinf(columns.reference.numeric).any()
        assert not np.isinf(columns.candidate.numeric).any()
        return from_columns(cls, columns, **kwargs)

    monkeypatch.setattr(TensorInputs, "from_columns", classmethod(tracked))
    result = _evaluate(
        real,
        generated,
        schema,
        seed=seed,
        options=options,
        metrics=_metric_instances(device, dtype),
        details=True,
    )
    # Complete-case geometry owns a filtered source domain; observed-value
    # fidelity and classifier missingness retain the original source pools.
    features = schema.numeric + schema.categorical
    complete_real = real.dropna(subset=features)
    complete_generated = generated.dropna(subset=features)
    assert transfers == [(len(real), len(generated)), (len(complete_real), len(complete_generated))]
    pd.testing.assert_frame_equal(real, originals[0])
    pd.testing.assert_frame_equal(generated, originals[1])
    sessions = {}
    for complete, source, candidate in [
        (False, real, generated),
        (True, complete_real, complete_generated),
    ]:
        draw = fixed_draw(len(source), len(candidate), seed=seed)
        columns = try_prepare(source, candidate, schema, retain_labels=True)
        sessions[complete] = (
            draw,
            ExecutionSession(
                source,
                candidate,
                schema,
                n_jobs=2,
                n_eff=draw.n_res,
                columns=columns,
                retain_labels=True,
                classifier_preprocessing=not complete,
            ),
        )
    for metric in _metric_instances(device, dtype):
        draw, session = sessions[metric.name == "prdc"]
        context = session.context_for(metric)
        actual = _metric(result, metric.name)
        assert result.provenance["metric_execution"][metric.name]["device"].startswith(
            device.split(":")[0]
        )
        for comparison, candidate, ideal, values in [
            (0, draw.generated, False, _values(actual)),
            (1, draw.real_candidate, True, _values(actual, reference=True)),
        ]:
            if metric.name == "linear_c2st":
                rf, sf, fit_seed = _classifier_protocol(metric, draw.n_res, seed, comparison)
                expected = context.compute_classifier(
                    metric,
                    draw.reference[None, :],
                    candidate[None, :],
                    rf[None, :],
                    sf[None, :],
                    [fit_seed],
                    ideal=ideal,
                    point=True,
                )[0]
                assert expected.converged
                np.testing.assert_allclose(values["auc"], expected.value, rtol=0, atol=1e-12)
                diagnostics = actual.diagnostics["point"][
                    "real_reference" if ideal else "generated"
                ]
                assert diagnostics["point"] is True
                assert diagnostics["fit_seed"] == fit_seed
                assert diagnostics["converged"] == expected.converged
                assert len(diagnostics["folds"]) == metric.cv_folds
            else:
                expected = context.compute(
                    metric, draw.reference[None, :], candidate[None, :], ideal=ideal
                )
                np.testing.assert_array_equal(
                    np.asarray([values[name] for name in metric.stat_names]).view(np.uint64),
                    np.asarray([expected[name][0] for name in metric.stat_names]).view(np.uint64),
                )
        if metric.name in {"shape", "trend", "conditional_fidelity"}:
            expected_details = context.compute_detailed(metric, draw.reference, draw.generated)
            assert actual.details is not None
            assert (
                list(actual.details.metadata["column_names"]) == schema.numeric + schema.categorical
            )
            for name, table in expected_details.details.tables.items():
                for column, expected in table.columns.items():
                    np.testing.assert_array_equal(
                        actual.details.tables[name].columns[column], expected
                    )
        else:
            assert actual.details is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fixed_cpu_pairs_match_explicit_backend_draws(dtype, monkeypatch):
    torch = pytest.importorskip("torch")
    _assert_backend_oracles(torch, "cpu", dtype, monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fixed_cuda_pairs_match_explicit_backend_draws(dtype, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    _assert_backend_oracles(torch, "cuda:0", dtype, monkeypatch)


def test_fixed_native_classifier_keeps_point_policy_and_failure_diagnostics(monkeypatch):
    pytest.importorskip("torch")
    from tabgen_eval.execution.torch.context import TorchContext

    real, generated, schema = _frames()
    calls = []

    def fit(self, metric, rd, sd, rf, sf, seeds, *, ideal=False, point=False, fit_threads=None):
        calls.append(dict(rd=rd.copy(), sd=sd.copy(), ideal=ideal, point=point))
        return [
            ClassifierOutcome(
                0.63 if ideal else 0.72,
                [dict(converged=False, reason="fixed native diagnostic marker", fold=0)],
            )
        ]

    monkeypatch.setattr(TorchContext, "compute_classifier", fit)
    result = _evaluate(
        real,
        generated,
        schema,
        metrics=["linear_c2st"],
        seed=1197,
        options=_config("cpu", "float64"),
    )
    result = _metric(result, "linear_c2st")
    assert len(calls) == 2
    assert all(call["point"] for call in calls)
    assert [call["ideal"] for call in calls] == [False, True]
    np.testing.assert_array_equal(calls[0]["rd"], calls[1]["rd"])
    assert not np.intersect1d(calls[1]["rd"], calls[1]["sd"]).size
    assert result.stats["auc"].value == 0.72
    assert result.stats["auc"].ideal == 0.63
    for comparison in ["generated", "real_reference"]:
        diagnostics = result.diagnostics["point"][comparison]
        assert diagnostics["converged"] is False
        assert diagnostics["folds"][0]["reason"] == "fixed native diagnostic marker"
