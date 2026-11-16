"""Training-referenced point scores use the shared report and export contract."""

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Report, Schema
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def test_memorization_report_keeps_geometry_diagnostics_and_unavailable_ratio():
    training = pd.DataFrame({"x": [1.0, 1.0, 1.0]})
    generated = pd.DataFrame({"x": [1.0, 2.0]})
    report = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=training.iloc[:2],
        options=make_options(uncertainty="both", n_replicates=10),
    )
    assert isinstance(report, Report)
    assert report["dcr/median"].value == 0.5
    assert np.isnan(report["dcr/ratio"].value)
    assert "ZERO_BASELINE" in report["dcr/ratio"].flag_codes
    assert report.results[0].diagnostics["holdout_median"] == 0.0
    for _, score in report.iter_stats():
        assert np.isnan(score.ideal)
        assert np.isnan(score.ci_low) and np.isnan(score.ci_high)
        assert score.generated_distribution is None
        assert score.reference_distribution is None
        assert not hasattr(score, "comparison")
    assert report.provenance["n_training"] == 3
    assert report.provenance["n_generated"] == 2
    assert report.provenance["n_selected_training"] == 3
    assert report.provenance["n_selected_generated"] == 2
    assert report.provenance["uncertainty"] == "none"
    assert report.provenance["n_replicates"] == 0
    payload = json.loads(report.to_json())
    ratio = payload["results"]["memorization"]["dcr"]["ratio"]
    assert ratio["value"] is None
    assert ratio["value_status"] == "nan"
    assert "comparison" not in ratio
    assert "ZERO_BASELINE" in ratio["flags"]
    assert payload["provenance"]["metric_diagnostics"]["dcr"]["holdout_median"] == 0.0
    assert list(report.to_dataframe()["stat"]) == [
        name for result in report.results for name in result.stats
    ]


def test_dcr_adapter_does_not_call_the_public_evaluator(monkeypatch):
    import tabgen_eval._evaluation.memorization as public
    from tabgen_eval.metrics import DCR

    def forbidden(*args, **kwargs):
        pytest.fail("numerical metric adapters must call execution directly")

    monkeypatch.setattr(public, "evaluate_memorization", forbidden)
    values = DCR().compute(
        pd.DataFrame({"x": [0.0, 2.0, 4.0]}),
        pd.DataFrame({"x": [1.0, 3.0]}),
        Schema(numeric=["x"]),
        rng=np.random.default_rng(9),
        holdout=pd.DataFrame({"x": [6.0]}),
    )
    assert values["median"] == 0.25
    assert values["ratio"] == 0.5
