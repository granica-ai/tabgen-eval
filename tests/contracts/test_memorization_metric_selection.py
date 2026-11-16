"""Exact DCR and optional near-match rate use explicit training roles."""

import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, evaluate, list_metrics
from tabgen_eval.metrics import DCR
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["training_coverage"])
def test_removed_metrics_are_unavailable_in_all_public_entry_points(name):
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    assert name not in {entry["name"] for entry in list_metrics()}
    schema = Schema(numeric=["x"])
    for roles in (
        {"test": frame, "generated": frame},
        {"test": frame},
        {"train": frame, "generated": frame},
    ):
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate(EvalData(**roles, schema=schema), metrics=[name])


@pytest.mark.parametrize("method", ["approximate", "auto"])
def test_dcr_rejects_removed_search_modes(method):
    with pytest.raises(ValueError, match="only supports search_method='exact'"):
        DCR(search_method=method)


@pytest.mark.parametrize(
    "option",
    [
        "projection_dim",
        "ivf_lists",
        "ivf_probes",
        "ivf_candidates",
        "cpu_ann_workspace_mb",
        "near_match_threshold",
    ],
)
def test_dcr_rejects_removed_search_options(option):
    with pytest.raises(TypeError, match="unexpected kwarg"):
        DCR(**{option: 1})


def test_explicit_memorization_metrics_remain_point_only():
    training = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
    generated = pd.DataFrame({"x": [1.0, 3.0]})
    holdout = pd.DataFrame({"x": [6.0]})
    result = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=holdout,
        metrics=["dcr"],
        options=make_options(uncertainty="both", n_replicates=99),
    )
    assert [m.name for m in result.results] == ["dcr"]
    assert result["dcr/median"].value == 0.25
    assert result["dcr/ratio"].value == 0.5
    assert result.provenance["metric_diagnostics"]["dcr"]["search_method"] == "exact"
    assert result.provenance["uncertainty"] == "none"
    assert result.provenance["n_replicates"] == 0
