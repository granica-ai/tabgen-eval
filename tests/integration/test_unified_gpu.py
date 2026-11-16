"""Unified public evaluation retains CPU/CUDA values and source draw streams."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, evaluate
from tests.support.options import make_options, public_kwargs, replace_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def test_unified_selected_metrics_and_uq_match_cpu_on_cuda(torch):
    from tests.support.backends import cuda_device

    rng = np.random.default_rng(813)

    def frame(size, shift=0):
        return pd.DataFrame(
            {
                "x": rng.normal(shift, 1, size),
                "y": rng.normal(size=size),
                "c": rng.choice(["a", "b", "c"], size, p=[0.7, 0.2, 0.1]),
            }
        )

    data = EvalData(
        train=frame(96),
        test=frame(100),
        generated=frame(80, 0.3),
        schema=Schema(numeric=["x", "y"], categorical=["c"]),
    )
    metrics = ["cond_shape", "recovery", "precision", "coverage", "dcr_ratio"]
    options = make_options(
        device="cpu",
        dtype="float64",
        seed=129,
        uncertainty="both",
        n_replicates=3,
    )
    cpu = evaluate(data, metrics=metrics, **public_kwargs(options), details=True)
    gpu = evaluate(
        data,
        metrics=metrics,
        **public_kwargs(replace_options(options, device=cuda_device(torch))),
        details=True,
    )
    assert [metric.name for metric in cpu.results] == [metric.name for metric in gpu.results]
    for result, expected in cpu.iter_stats():
        actual = gpu[f"{result.name}/{expected.name}"]
        assert actual.score_role == expected.score_role
        assert actual.ideal_status == expected.ideal_status
        assert actual.point_status == expected.point_status
        for field in ("value", "ideal", "ci_low", "ci_high", "ideal_ci_low", "ideal_ci_high"):
            np.testing.assert_allclose(
                getattr(actual, field),
                getattr(expected, field),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        for field in ("generated_distribution", "reference_distribution"):
            a, b = getattr(actual, field), getattr(expected, field)
            if b is None:
                assert a is None
            else:
                assert a is not None
                assert a.requested_count == b.requested_count == 3
                np.testing.assert_allclose(
                    a.samples, b.samples, rtol=1e-12, atol=1e-12, equal_nan=True
                )
    assert gpu["cond_shape"] is gpu["cond_shape/cond_shape_all"]
    assert gpu["recovery"] is gpu["recovery/recovery_all"]
    assert gpu["precision"] is gpu["precision/precision"]
    assert gpu["coverage"] is gpu["coverage/coverage"]
    assert gpu["dcr_ratio"] is gpu["dcr_ratio/ratio"]
    assert gpu["dcr_ratio"].score_role == "distance_ratio"
    assert gpu["dcr_ratio/median"].score_role == "generated_train"
    assert gpu["dcr_ratio/holdout_q50"].score_role == "test_train"
    assert all(stat.ideal_status == "absent" for stat in gpu.results[-1].stats.values())
    assert gpu["cond_shape"].score_role == "generated_test"
    assert gpu.provenance["input_roles"] == ["generated", "test", "train"]
    assert gpu.provenance["procedures"]["memorization"]["uncertainty"] == "none"
    assert all(
        options["device"].startswith("cuda")
        for options in gpu.provenance["metric_execution"].values()
    )


def test_memorization_inference_does_not_change_with_comparison_selection(torch):
    from tests.support.backends import cuda_device

    rng = np.random.default_rng(711)
    train = pd.DataFrame({"x": np.arange(96), "y": rng.normal(size=96)})
    test = pd.DataFrame({"x": np.tile(np.arange(8), 12), "y": rng.normal(size=96)})
    generated = pd.DataFrame({"x": rng.integers(96, size=80), "y": rng.normal(size=80)})
    data = EvalData(train=train, test=test, generated=generated, schema=Schema.infer(train))
    options = make_options(device=cuda_device(torch), seed=119)
    alone = evaluate(data, metrics=["dcr_ratio"], **public_kwargs(options))
    combined = evaluate(data, metrics=["shape", "dcr_ratio"], **public_kwargs(options))
    for name, stat in alone.results[0].stats.items():
        np.testing.assert_equal(combined[f"dcr_ratio/{name}"].value, stat.value)
    assert (
        alone.provenance["procedures"]["memorization"]["schema"]
        == combined.provenance["procedures"]["memorization"]["schema"]
    )
