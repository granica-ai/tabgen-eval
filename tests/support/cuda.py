"""Launch packaged CUDA device kernels directly for independent numerical checks."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any


def launch_kernel(
    ptx: str | Path,
    name: str,
    *,
    blocks: int,
    threads: int,
    arguments: list[Any],
    stream: int,
) -> None:
    """Launch a production PTX entry point on the caller's current CUDA context.

    Tensor owners and host scalar arguments remain alive until launch completion.
    Synchronize the supplied stream before unloading the private driver module;
    this keeps the check independent of the package's optimizer wrapper.
    """
    driver = ctypes.CDLL("libcuda.so.1")
    module, kernel = ctypes.c_void_p(), ctypes.c_void_p()

    def checked(function: str, *args: Any) -> None:
        status = getattr(driver, function)(*args)
        if status:
            raise RuntimeError(f"{function} failed with CUDA driver status {status}")

    checked("cuModuleLoad", ctypes.byref(module), str(ptx).encode())
    try:
        checked("cuModuleGetFunction", ctypes.byref(kernel), module, name.encode())
        pointers = (ctypes.c_void_p * len(arguments))(
            *(ctypes.cast(ctypes.byref(argument), ctypes.c_void_p) for argument in arguments)
        )
        checked(
            "cuLaunchKernel",
            kernel,
            blocks,
            1,
            1,
            threads,
            1,
            1,
            0,
            ctypes.c_void_p(stream),
            pointers,
            None,
        )
        checked("cuStreamSynchronize", ctypes.c_void_p(stream))
    finally:
        checked("cuModuleUnload", module)
