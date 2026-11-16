"""Small runnable examples exercise the public API without network access."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "script,arguments",
    [
        ("quickstart.py", ["--rows", "40", "--device", "cpu"]),
        ("report_walkthrough.py", []),
        ("visualization_walkthrough.py", []),
        (
            "benchmark_backends.py",
            [
                "--rows",
                "40",
                "--columns",
                "3",
                "--categorical",
                "1",
                "--repeat",
                "1",
                "--devices",
                "cpu",
            ],
        ),
    ],
)
def test_example_cli_smoke(script, arguments, tmp_path, checkout_root):
    import tabgen_eval

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(tabgen_eval.__file__).resolve().parents[1])
    if script == "benchmark_backends.py":
        arguments = [*arguments, "--out", str(tmp_path / "benchmark.json")]
    if script == "visualization_walkthrough.py":
        arguments = [*arguments, "--output", str(tmp_path / "visualization")]
    result = subprocess.run(
        [sys.executable, str(checkout_root / "examples" / script), *arguments],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout.strip()
    if script == "visualization_walkthrough.py":
        output = tmp_path / "visualization"
        for name in (
            "scores",
            "columns",
            "pairs",
            "slices",
            "concentration",
            "prdc",
            "association",
        ):
            assert 'src="plotly.min.js"' in (output / f"{name}.html").read_text()
        assert (output / "plotly.min.js").is_file()
        artifact = json.loads((output / "report.json").read_text())
        concentration = artifact["results"]["concentration"]["concentration"]
        assert "mean_radii_1e-3" in concentration
        assert "upper_radii_1e-3" not in concentration
    if (tmp_path / "benchmark.json").exists():
        artifact = json.loads((tmp_path / "benchmark.json").read_text())
        assert artifact
        if script == "benchmark_backends.py":
            assert artifact["settings"]["uncertainty"] is None
            assert artifact["inputs"]["n_res"] == 20
            assert artifact["hardware"]["cpu"]["model"]
            cases = {(case["metric"], case["case"]): case for case in artifact["cases"]}
            assert len(cases) == 4
            for metric in ("shape", "trend", "conditional_fidelity", "linear_c2st"):
                native = cases[metric, "cpu"]
                assert native["execution"]["implementation"] == "aten_cpp"
                assert native["n_res"] == 20
                assert native["status"] == "ok"
                assert native["first_call_s"] > 0
                assert len(native["warm_runs_s"]) == 1 and native["warm_median_s"] > 0
                assert native["scores"] == native["first_call_scores"]
                if metric == "linear_c2st":
                    point = native["classifier_diagnostics"]["point"]
                    assert point["generated"]["converged"] is True
                    assert all(fold["solver"] for fold in point["generated"]["folds"])


def test_openml_tutorial_uses_supplied_loader_without_network(monkeypatch, checkout_root, capsys):
    path = checkout_root / "examples" / "openml_quickstart.py"
    spec = importlib.util.spec_from_file_location("openml_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = [
        pd.DataFrame({"x": np.arange(40, dtype=float), "c": ["a", "b"] * 20}) for _ in range(2)
    ]
    calls = []

    def load(identifier):
        calls.append(identifier)
        return frames

    monkeypatch.setattr(module, "load_openml", load)
    module.main(["179", "--n-replicates", "2", "--device", "cpu"])
    assert calls == [179]
    assert "n_res=20" in capsys.readouterr().out


def test_backend_benchmark_supplied_files_and_native_precision(tmp_path, checkout_root):
    import tabgen_eval

    holdout = pd.DataFrame({"x": np.arange(24, dtype=float), "c": ["a", "b"] * 12})
    generated = holdout.iloc[:16].copy()
    holdout.to_csv(tmp_path / "real.csv", index=False)
    generated.to_csv(tmp_path / "generated.csv", index=False)
    (tmp_path / "schema.json").write_text(json.dumps({"numeric": ["x"], "categorical": ["c"]}))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(tabgen_eval.__file__).resolve().parents[1])
    subprocess.run(
        [
            sys.executable,
            str(checkout_root / "examples" / "benchmark_backends.py"),
            "--holdout",
            str(tmp_path / "real.csv"),
            "--generated",
            str(tmp_path / "generated.csv"),
            "--schema",
            str(tmp_path / "schema.json"),
            "--devices",
            "cpu",
            "--dtype",
            "float32",
            "--metrics",
            "shape",
            "--repeat",
            "1",
            "--out",
            str(tmp_path / "artifact.json"),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert artifact["inputs"]["n_holdout"] == 24
    assert artifact["inputs"]["n_generated"] == 16
    assert artifact["inputs"]["n_res"] == 12
    assert artifact["libraries"]["tabgen-eval"]
    (native,) = artifact["cases"]
    assert native["actual_numeric_dtype"] == native["execution"]["dtype"] == "float32"
