"""Supported device interchange without copying feature matrices to the host."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from tabgen_eval._typing import FloatArray, IntArray

from .types import CudaArray

if TYPE_CHECKING:
    from torch import Tensor

    from .types import ModelInput


def _is_tensor(value: ModelInput) -> TypeGuard[Tensor]:
    """Recognize tensor storage without importing Torch for NumPy-only callers."""
    return type(value).__module__.startswith("torch")


def engine_array(value: ModelInput) -> FloatArray | IntArray | CudaArray:
    """Expose Torch buffers as NumPy/CuPy arrays, retaining their storage owners.

    CuPy consumes Torch's DLPack producer stream and owns the resulting view for
    the duration of XGBoost's synchronous Python calls. XGBoost's supported CUDA
    array input path itself requires CuPy, even for custom array-interface views.
    Imports remain lazy so CPU XGBoost does not require Torch or CUDA packages.
    """
    if not _is_tensor(value):
        return value
    tensor = value.detach()
    if not tensor.is_contiguous():
        raise ValueError("XGBoost preprocessing must produce contiguous tensor buffers")
    if tensor.device.type == "cpu":
        return tensor.numpy()
    if tensor.device.type != "cuda":
        raise ValueError("XGBoost supports CPU and CUDA preprocessing buffers")
    try:
        import cupy
    except ModuleNotFoundError as error:
        if error.name != "cupy":
            raise
        raise ModuleNotFoundError(
            "CUDA XGBoost tensor input requires CuPy; install 'tabgen-eval[xgboost-cuda]' "
            "or the compatible CuPy package for your CUDA installation.",
            name="cupy",
        ) from error
    with cupy.cuda.Device(tensor.device.index):
        return cupy.from_dlpack(tensor)
