"""CPU operator fixtures; loaded once through the normal extension boundary."""

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def ops(torch):
    from tabgen_eval.execution.torch.cpu.extension import library

    return library()
