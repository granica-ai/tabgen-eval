"""Explicit backend/precision cases and consistent CUDA availability checks."""

import pytest


def cuda_device(torch):
    """Select the isolated visible GPU, preferring device one on shared machines."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(torch, request):
    return "cpu" if request.param == "cpu" else cuda_device(torch)


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        ("cpu", "float32"),
        ("cpu", "float64"),
        pytest.param(("cuda", "float32"), marks=pytest.mark.gpu),
        pytest.param(("cuda", "float64"), marks=pytest.mark.gpu),
    ]
)
def backend_precision(request):
    backend, precision = request.param
    module = pytest.importorskip("torch")
    if backend == "cuda":
        backend = cuda_device(module)
    return backend, precision


@pytest.fixture
def backend(backend_precision):
    return backend_precision[0]


@pytest.fixture
def backend_dtype(backend_precision):
    return backend_precision[1]
