"""Native rate-policy invariants, independent of CUDA or matrix factorization."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def native_rate(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    directory = tmp_path_factory.mktemp("saga_rate")
    from tabgen_eval.execution.torch import binary

    source = Path(__file__).with_name("saga_rate_probe.cpp")
    executable = directory / "rate"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(binary.ROOT / "csrc"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )

    def calculate(rows):
        output = subprocess.run(
            [str(executable)],
            input="\n".join(" ".join(map(str, row)) for row in rows) + "\n",
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        return output.splitlines()

    return calculate


def test_single_sample_and_full_batch_endpoints(native_rate):
    single, full = [
        np.fromstring(row, sep=" ")
        for row in native_rate([(1000, 1, 1000, 3, 100, 0.5), (1000, 1000, 1000, 3, 100, 0.5)])
    ]
    assert single[0] == pytest.approx(1 / (200 + 0.5))
    assert full[0] == pytest.approx(1 / (6 + 0.5))
    assert single[1] == 200
    assert full[1] == 6


def test_smaller_batches_receive_no_larger_rates(native_rate):
    batches = [1, 16, 256, 1024, 4096, 1000000]
    rows = [
        np.fromstring(row, sep=" ")
        for row in native_rate([(1000000, b, 4096, 3, 100, 0.5) for b in batches])
    ]
    assert np.all(np.diff([row[0] for row in rows]) >= 0)


def test_actual_batch_clamps_to_population_and_one_row_is_finite(native_rate):
    rows = native_rate([(1, 4096, 1, 3, 3, 0.5), (5, 4096, 5, 3, 3, 0.5)])
    values = [np.fromstring(row, sep=" ") for row in rows]
    assert values[0][0] == values[1][0] == pytest.approx(1 / 6.5)
    assert [v[2] for v in values] == [1, 5]


def test_ridge_is_added_separately_and_can_be_zero(native_rate):
    rows = [
        np.fromstring(row, sep=" ")
        for row in native_rate([(500000, 512, 4096, 3, 100, ridge) for ridge in (0, 0.5, 10)])
    ]
    assert rows[0][0] > rows[1][0] > rows[2][0]
    assert all(row[1] == rows[0][1] for row in rows)


def test_invalid_dimensions_and_curvatures_are_rejected(native_rate):
    assert (
        native_rate(
            [
                (0, 1, 1, 3, 100, 0.5),
                (5, 0, 1, 3, 100, 0.5),
                (5, 1, 6, 3, 100, 0.5),
                (5, 1, 1, 0, 100, 0.5),
                (5, 1, 1, 3, -1, 0.5),
                (5, 1, 1, 3, 100, -1),
            ]
        )
        == ["invalid"] * 6
    )
