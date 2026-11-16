"""User-supplied temporal conversion retains metric meanings across backends."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def test_mixed_datetime_units_preserve_shape_and_dcr(backend_precision):
    backend, precision = backend_precision
    options = make_options(
        device="cpu" if backend == "python" else backend,
        dtype=precision,
        seed=19,
        n_jobs=4,
    )
    schema = Schema(numeric=["t"])
    dates = np.arange("2020-01-01", "2020-01-17", dtype="datetime64[D]")
    training = pd.DataFrame({"t": dates.astype("datetime64[ns]")})
    draw = fixed_draw(len(training), len(training) // 2, seed=options["seed"])
    generated = pd.DataFrame({"t": dates[draw.reference].astype("datetime64[ms]")})
    holdout = pd.DataFrame({"t": (dates + np.timedelta64(40, "D")).astype("datetime64[s]")})
    # The caller chooses a common unit before evaluation.
    for frame in (training, generated, holdout):
        frame["t"] = frame.t.astype("datetime64[ms]").astype("int64").astype(float)
    shape = evaluate(training, generated, schema, metrics=["shape"], options=options)
    memorization = evaluate_memorization(
        training, generated, schema, holdout=holdout, metrics=["dcr"], options=options
    )
    assert shape["shape/shape"].value == 1.0
    for stat in ["median", "q05", "ratio"]:
        assert memorization[f"dcr/{stat}"].value == 0.0
    assert memorization.provenance["metric_diagnostics"]["dcr"]["holdout_median"] > 0
