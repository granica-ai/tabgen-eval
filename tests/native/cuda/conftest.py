"""CUDA availability, device precision and operator fixtures."""

import pytest

from tests.support.backends import cuda_device


@pytest.fixture
def torch():
    module = pytest.importorskip("torch")
    if not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    return module


@pytest.fixture
def device(torch):
    return cuda_device(torch)


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    return request.param


@pytest.fixture
def ops(torch):
    from tabgen_eval.execution.torch.cuda.extension import library

    return library()
