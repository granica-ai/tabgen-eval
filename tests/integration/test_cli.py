"""CLI smoke test: load CSVs with a schema, run, emit a report."""

from __future__ import annotations

import inspect
import json

import pytest

from tabgen_eval import evaluate
from tabgen_eval.cli import _build_parser, main
from tests.helpers import make_frame

pytestmark = pytest.mark.unit


def test_cli_mirrors_evaluate_arguments_and_defaults():
    parser = _build_parser()
    flags = {action.dest: action for action in parser._actions}
    for name, parameter in inspect.signature(evaluate).parameters.items():
        if name != "data":
            assert name in flags, f"evaluate argument {name} has no corresponding CLI flag"
        if name in {"seed", "device", "dtype", "n_jobs", "details"}:
            assert flags[name].default == parameter.default


@pytest.mark.parametrize("value", ["auto", "torch", "python"])
def test_cli_rejects_removed_backend_parameter(value):
    parser = _build_parser()
    assert all(action.dest != "backend" for action in parser._actions)
    with pytest.raises(SystemExit):
        parser.parse_args(["--schema", "schema.json", "--backend", value])


def test_cli_rejects_removed_reference_cache_option():
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(["--schema", "schema.json", "--reference-cache", "cache"])
    assert error.value.code == 2


@pytest.mark.parametrize("include_replicates", [False, True])
@pytest.mark.parametrize("mode", ["candidate", "both"])
def test_cli_end_to_end(tmp_path, capsys, include_replicates, mode):
    test = make_frame(300, seed=1)
    syn = make_frame(250, seed=2)
    test.to_csv(tmp_path / "test.csv", index=False)
    syn.to_csv(tmp_path / "syn.csv", index=False)
    schema = {"numeric": ["age", "income"], "categorical": ["job", "marital"]}
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    rc = main(
        [
            "--holdout",
            str(tmp_path / "test.csv"),
            "--generated",
            str(tmp_path / "syn.csv"),
            "--schema",
            str(tmp_path / "schema.json"),
            "--metrics",
            "shape",
            "trend",
            "--uncertainty",
            mode,
            "--n-replicates",
            "3",
            "--out",
            str(tmp_path / "report.json"),
        ]
        + (["--include-replicates"] if include_replicates else [])
    )
    assert rc == 0
    data = json.loads((tmp_path / "report.json").read_text())
    assert "results" in data
    assert "fidelity" in data["results"]
    score = data["results"]["fidelity"]["shape"]["shape"]
    for key in ("generated_distribution", "reference_distribution"):
        if mode == "candidate" and key == "reference_distribution":
            assert score[key] is None
            continue
        assert score[key]["requested_count"] == 3
        assert ("samples" in score[key]) == include_replicates
    assert "config" not in data
    assert data["options"]["uncertainty"] == {"mode": mode, "n_replicates": 3}
    assert "n_replicates" not in data["options"]
    assert "uq_config" not in data["provenance"]


@pytest.mark.parametrize("mode", ["reference", "candidate", "both"])
def test_cli_exposes_shared_repetition_options_and_rejects_bootstrap(mode):
    parser = _build_parser()
    required = ["--holdout", "a.csv", "--generated", "b.csv", "--schema", "schema.json"]
    args = parser.parse_args(required + ["--uncertainty", mode, "--n-replicates", "2"])
    assert args.uncertainty == mode and args.n_replicates == 2
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--n-bootstrap", "2"])
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--alpha", "0.1"])


@pytest.mark.parametrize("count", ["0", "-1", "1.5", "True"])
def test_cli_rejects_invalid_replicate_counts(count, capsys):
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(["--schema", "schema.json", "--n-replicates", count])
    assert error.value.code == 2
    assert "--n-replicates" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option", ["--metric-kwargs", "--gpu-workspace-mb", "--memory-limit-mb", "--workspace-mb"]
)
def test_cli_rejects_removed_metric_and_memory_options(option):
    value = "{}" if option == "--metric-kwargs" else "32"
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--test", "a.csv", option, value])


def test_cli_json_stdout_excludes_evaldata_metric_summary(tmp_path, capsys):
    make_frame(80, seed=11).to_csv(tmp_path / "test.csv", index=False)
    make_frame(80, seed=12).to_csv(tmp_path / "generated.csv", index=False)
    (tmp_path / "schema.json").write_text(
        json.dumps({"numeric": ["age", "income"], "categorical": ["job", "marital"]})
    )

    assert (
        main(
            [
                "--test",
                str(tmp_path / "test.csv"),
                "--generated",
                str(tmp_path / "generated.csv"),
                "--schema",
                str(tmp_path / "schema.json"),
                "--metrics",
                "shape",
                "--format",
                "json",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "shape" in payload["results"]["fidelity"]
    assert captured.err.startswith("EvalData (test + generated) — default evaluate metrics:\n")
    assert "  fidelity: shape, trend\n" in captured.err
    assert captured.err.count("default evaluate metrics:") == 1


def test_cli_mle_repetitions_export_raw_scores(tmp_path):
    pytest.importorskip("xgboost", minversion="2.1")
    real = make_frame(300, seed=8)
    generated = make_frame(250, seed=9)
    real.to_csv(tmp_path / "real.csv", index=False)
    generated.to_csv(tmp_path / "generated.csv", index=False)
    (tmp_path / "schema.json").write_text(
        json.dumps(
            {
                "numeric": ["age", "income"],
                "categorical": ["job", "marital"],
                "target": "income",
            }
        )
    )
    assert (
        main(
            [
                "--holdout",
                str(tmp_path / "real.csv"),
                "--generated",
                str(tmp_path / "generated.csv"),
                "--schema",
                str(tmp_path / "schema.json"),
                "--metrics",
                "mle",
                "--uncertainty",
                "both",
                "--n-replicates",
                "2",
                "--include-replicates",
                "--dtype",
                "float32",
                "--device",
                "cpu",
                "--out",
                str(tmp_path / "mle.json"),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "mle.json").read_text())
    score = payload["results"]["utility"]["mle"]["score"]
    assert score["direction"] == "lower_better"
    ratio = payload["results"]["utility"]["mle"]["ratio"]
    assert ratio["direction"] == "lower_better"
    assert ratio["value"] == pytest.approx(score["value"] / score["ideal"])
    assert len(score["generated_distribution"]["samples"]) == 2
    assert len(score["reference_distribution"]["samples"]) == 2
    assert "interval_method" not in score
    assert payload["provenance"]["metric_execution"]["mle"]["device"] == "cpu"
