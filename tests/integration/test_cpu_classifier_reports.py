"""Failed native CPU fits retain report flags and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import LinearC2ST
from tests.support.evaluation import paired_report as evaluate
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("memory_failure", [False, True])
def test_compiled_classifier_report_retains_failed_fit_flags_and_diagnostics(torch, memory_failure):
    rng = np.random.default_rng(44)
    width = 200 if memory_failure else 3
    columns = [f"x{i}" for i in range(width)]
    r = pd.DataFrame(rng.normal(size=(240, width)), columns=columns)
    s = pd.DataFrame(rng.normal(2.0, 1.0, size=(240, width)), columns=columns)
    metric_options = {"memory_limit_mb": 1} if memory_failure else {"max_iter": 1}
    options = make_options(device="cpu", uncertainty="both", n_replicates=3)
    report = evaluate(
        r,
        s,
        Schema(numeric=columns),
        metrics=[
            configured_metric(
                LinearC2ST,
                options,
                **metric_options,
            )
        ],
        options=options,
    )
    assert "CLASSIFIER_NOT_CONVERGED" in report["linear_c2st/auc"].flag_codes
    diagnostic = report.provenance["classifier_diagnostics"]["linear_c2st"]
    repeated = diagnostic["replicates"]["generated"]
    assert len(repeated) == 3 and all(not entry["converged"] for entry in repeated)
    distribution = report["linear_c2st/auc"].generated_distribution
    assert distribution.finite_count == (0 if memory_failure else 3)
    assert distribution.unavailable_count == (3 if memory_failure else 0)
    reasons = (
        {"preprocessing_or_memory_limit", "saga_memory_limit"} if memory_failure else {"max_iter"}
    )
    assert all(
        f["reason"] in reasons and not f["converged"] for fit in repeated for f in fit["folds"]
    )
