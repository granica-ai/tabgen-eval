"""Intrinsic Conditional applicability is shared by paired and calibrated evaluation."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import Conditional
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2])
def test_narrow_conditional_protocols_preserve_undefined_statistics(width):
    pytest.importorskip("torch")
    names = [f"c{i}" for i in range(width)]
    real = pd.DataFrame({name: ["real"] * 24 for name in names})
    generated = pd.DataFrame({name: ["generated"] * 12 for name in names})
    # Physical table width does not determine the number of modeled axes.
    real["unused"] = np.arange(len(real))
    generated["unused"] = np.arange(len(generated))
    schema = Schema(categorical=names)
    options = make_options(
        device="cpu",
    )
    fixed_report = evaluate(
        real,
        generated,
        schema,
        metrics=[configured_metric(Conditional, options, min_count=1)],
        options=options,
        details=True,
    )
    fixed = next(result for result in fixed_report.results if result.name == "conditional_fidelity")
    calibrated = calibrate(
        real,
        schema,
        metrics=[configured_metric(Conditional, options, min_count=1)],
        n_replicates=3,
        options=options,
    ).metrics["conditional_fidelity"]

    undefined = set(Conditional().stat_names)
    defined = {"recovery_all"}
    if width == 2:
        defined.add("cond_shape_all")
    undefined -= defined
    for name in defined:
        assert fixed.stats[name].value == 0.0
        assert fixed.stats[name].ideal == 1.0
        np.testing.assert_array_equal(summaries(calibrated)[name].samples, np.ones(3))
    for name in undefined:
        assert np.isnan(fixed.stats[name].value)
        assert np.isnan(fixed.stats[name].ideal)
        statistic = summaries(calibrated)[name]
        assert np.isnan(statistic.samples).all()
        assert statistic.finite_count == 0
        assert statistic.unavailable_count == 3

    columns = fixed.details.tables["columns"].columns
    slices = fixed.details.tables["slices"].columns
    assert len(columns["column_index"]) == width
    assert len(slices["column_index"]) == width
    for name in undefined:
        assert np.isnan(columns[name]).all()
    assert np.isnan(slices["trend"]).all()
    if width == 1:
        assert np.isnan(slices["shape"]).all()
    else:
        np.testing.assert_array_equal(slices["shape"], np.zeros(width))
    np.testing.assert_array_equal(slices["recovery"], np.zeros(width))
