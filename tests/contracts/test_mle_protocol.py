"""Retained predictive protocol: shared draws, cached models, and private seeds."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.sampling import fixed_draw, paired_draw_batches
from tabgen_eval.models import XGBoostModel
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["none", "reference", "candidate", "both"])
def test_utility_plot_uses_reported_estimates_without_refitting(engine, monkeypatch, mode):
    real, generated, schema = pools()
    report = evaluate(
        real,
        generated,
        schema,
        metrics=["mle"],
        options=make_options(uncertainty=mode, n_replicates=3),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Plotting must not fit models or compute predictions")

    monkeypatch.setattr(XGBoostModel, "fit", forbidden)
    monkeypatch.setattr(XGBoostModel, "predict", forbidden)
    figure = report.utility.plot(kind="utility")
    assert len(figure.data) == 4
    assert list(figure.data[0].x) == [report["mle/score"].value]
    assert list(figure.data[1].x) == [report["mle/score"].ideal]
    assert list(figure.data[2].x) == [report["mle/ratio"].value]
    assert list(figure.data[3].x) == [report["mle/prediction_mse"].value]
    assert "MAE" in figure.layout.annotations[0].text
    assert "mle · y" in figure.layout.annotations[0].text
    if mode != "none":
        assert not report.results[0].diagnostics["point"]
    if mode in {"candidate", "both"}:
        assert report["mle/ratio"].value != pytest.approx(
            report["mle/score"].value / report["mle/score"].ideal
        )


@pytest.fixture
def engine(monkeypatch):
    fits, predictions = [], []
    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: None))

    def fit(self, train, labels, *, seed, fit_threads, **kwargs):
        assert tuple(kwargs["feature_names"]) == ("x", "id")
        assert tuple(kwargs["feature_types"]) == ("q", "q")
        assert tuple(kwargs["categorical_cardinalities"]) == ()
        self.identifier = len(fits)
        fits.append((train.cpu().numpy().copy(), seed, fit_threads))
        self.diagnostics = {"converged": True, "random_state": seed}
        return self

    def predict(self, test, *, prediction):
        assert prediction == "value"
        values = test.cpu().numpy().copy()
        predictions.append((self.identifier, values))
        return values[:, 0] + 0.01 * self.identifier

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    return fits, predictions


def pools(n_generated=37):
    real = pd.DataFrame({"x": np.linspace(-2, 2, 51), "id": np.arange(51)})
    generated = pd.DataFrame(
        {"x": np.linspace(-2, 2, n_generated), "id": -np.arange(1, n_generated + 1)}
    )
    return (
        real.assign(y=real.x),
        generated.assign(y=generated.x),
        Schema(numeric=["x", "id", "y"], target="y"),
    )


@pytest.mark.parametrize(
    "mode,expected_fits", [("none", 2), ("reference", 4), ("candidate", 6), ("both", 6)]
)
@pytest.mark.parametrize(
    "generated_size,fixed_res,both_res", [(13, 13, 6), (25, 25, 12), (37, 25, 25)]
)
def test_refits_and_rescoring_follow_source_protocol(
    engine, mode, expected_fits, generated_size, fixed_res, both_res
):
    real, generated, schema = pools(generated_size)
    options = make_options(uncertainty=mode, n_replicates=3, seed=43)
    report = evaluate(real, generated, schema, metrics=["mle"], options=options)
    fits, predictions = engine
    assert len(fits) == expected_fits
    n_res = both_res if mode in {"candidate", "both"} else fixed_res
    assert all(len(training) == n_res and threads == 1 for training, _, threads in fits)
    assert report["mle/score"].n_eff == n_res
    assert report.provenance["per_metric_sampling"]["mle"]["n_res"] == n_res
    sampling = report.results[0].diagnostics["sampling"]
    assert sampling["n_res"] == n_res
    assert [
        sampling[f"n_selected_{role}"] for role in ("generated", "real_train", "real_test")
    ] == [n_res] * 3
    assert len(predictions) == (2 if mode == "none" else 6)
    generated_models = []
    for generated_call, real_call in zip(predictions[::2], predictions[1::2], strict=True):
        assert len(generated_call[1]) == len(real_call[1]) == n_res
        np.testing.assert_array_equal(generated_call[1], real_call[1])
        assert set(fits[real_call[0]][0][:, 1]).isdisjoint(real_call[1][:, 1])
        generated_models.append(generated_call[0])
    if mode == "reference":
        assert generated_models == [0] * 3
    elif mode in {"candidate", "both"}:
        assert generated_models == [0, 2, 4]
        assert len({tuple(fits[index][0][:, 1]) for index in generated_models}) == 3
    for name in ("score", "ratio", "prediction_mse"):
        assert (report[f"mle/{name}"].generated_distribution is None) == (mode == "none")
    errors = [
        np.mean(np.abs(values[:, 0] - (values[:, 0] + 0.01 * identifier)))
        for identifier, values in predictions
    ]
    assert report["mle/score"].value == pytest.approx(np.mean(errors[::2]))
    assert report["mle/score"].ideal == pytest.approx(np.mean(errors[1::2]))
    assert report["mle/ratio"].value == pytest.approx(
        np.mean(np.asarray(errors[::2]) / errors[1::2])
    )
    if mode != "none":
        np.testing.assert_array_equal(
            report["mle/score"].generated_distribution.samples, errors[::2]
        )
        if mode in {"reference", "both"}:
            np.testing.assert_array_equal(
                report["mle/score"].reference_distribution.samples, errors[1::2]
            )
        else:
            assert report["mle/score"].reference_distribution is None
            assert np.isnan(report["mle/score"].ideal_ci_low)
        np.testing.assert_array_equal(
            report["mle/ratio"].generated_distribution.samples,
            np.asarray(errors[::2]) / errors[1::2],
        )
    assert report["mle/ratio"].reference_distribution is None
    assert np.isnan(report["mle/ratio"].ideal)


@pytest.mark.parametrize("mode", ["reference", "candidate", "both"])
@pytest.mark.parametrize("generated_size", [13, 37])
def test_source_seed_domains_and_full_input_draws_are_preserved(engine, mode, generated_size):
    real, generated, schema = pools(generated_size)
    options = make_options(seed=81, uncertainty=mode, n_replicates=2)
    evaluate(real, generated, schema, metrics=["mle"], options=options)
    fits, predictions = engine
    point = fixed_draw(len(real), len(generated), seed=81, uncertainty=mode)
    expected = [(point.reference, point.real_candidate, point.generated)] if mode == "none" else []
    for batch in paired_draw_batches(
        point,
        n_replicates=2,
        seed=81,
        batch_size=1,
        include_generated=mode in {"candidate", "both"},
    ):
        expected.append(
            (
                batch.reference[0],
                batch.real_candidate[0],
                point.generated if batch.generated is None else batch.generated[0],
            )
        )
    assert max(np.concatenate([point.reference, point.real_candidate])) >= 20

    def model_seed(replicate, generated_role):
        sequence = np.random.SeedSequence(81, spawn_key=(0x505252, replicate + 1, generated_role))
        return int(np.random.default_rng(sequence).integers(0, 2**32 - 1))

    for index, (reference, training, candidate) in enumerate(expected):
        generated_call, real_call = predictions[2 * index : 2 * index + 2]
        generated_fit, real_fit = fits[generated_call[0]], fits[real_call[0]]
        np.testing.assert_array_equal(generated_fit[0][:, 1], generated.iloc[candidate].id)
        np.testing.assert_array_equal(real_fit[0][:, 1], real.iloc[training].id)
        np.testing.assert_array_equal(real_call[1][:, 1], real.iloc[reference].id)
        assert generated_fit[1] == model_seed(index if mode in {"candidate", "both"} else -1, 1)
        assert real_fit[1] == model_seed(-1 if mode == "none" else index, 0)
