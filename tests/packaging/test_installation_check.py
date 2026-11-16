"""Exercise the real installation checker through its smallest supported path."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("block_torch", [False, True])
def test_installation_checker_runs_current_public_apis_outside_checkout(
    tmp_path, checkout_root, block_torch
):
    output = tmp_path / "installation.json"
    env = os.environ.copy()
    # Source checkouts use src/, installed-artifact CI supplies the wheel instead.
    import tabgen_eval

    env["PYTHONPATH"] = str(__import__("pathlib").Path(tabgen_eval.__file__).resolve().parents[1])
    subprocess.run(
        [
            sys.executable,
            str(checkout_root / "tools" / "check_installation.py"),
            "--smoke",
            "--out",
            str(output),
        ]
        + (["--block-torch"] if block_torch else []),
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(output.read_text())
    assert result["custom_metric"] == "passed"
    assert result["device"] == "cpu"
    assert result["classifier_missing_torch_rejection"] == ("passed" if block_torch else None)
    if block_torch:
        assert result["reports"] == []
        return
    selected = result["reports"][0]["report"]["provenance"]["metric_execution"]
    assert ("linear_c2st" in selected) is not block_torch
    assert [entry["case"] for entry in result["reports"]] == ["torch", "torch", "memorization"]
    fixed, repeated, memory = [entry["report"] for entry in result["reports"]]
    assert fixed["provenance"]["uncertainty"] == "none"
    assert repeated["provenance"]["uncertainty"] == "both"
    assert memory["provenance"]["reference"] == "selected_training"
    assert all(report["schema_version"] == 2 for report in (fixed, repeated, memory))
    assert {"precision", "coverage", "cond_shape", "recovery"} <= selected.keys()
    assert {"prdc", "conditional_fidelity", "sanity"}.isdisjoint(selected)
    assert set(memory["results"]["memorization"]) == {
        "dcr_ratio",
        "near_match_rate",
        "replication_ratio",
    }
