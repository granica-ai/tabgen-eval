"""Process-safe NVRTC device cache for compiled CUDA classifier optimizers.

The regular host extension/prebuilt library owns C++ operators; this adapter
compiles standalone header-free CUDA optimizer sources into PTX for a selected
GPU capability and FP32/FP64 precision. It does not request fast math or fused
multiply-add. Disk files are protected by a process lock and atomic replacement;
a thread lock protects per-device/dtype owners in this process. Native setup
retains loaded modules for process lifetime on the selected device.

Importing this module does not import Torch or initialize a CUDA context. The
first ``operators`` call loads Torch, the host library, and the selected module.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal, TypeAlias

from tabgen_eval.execution.torch.cuda.extension import library
from tabgen_eval.execution.torch.sources import device_source

if TYPE_CHECKING:
    from torch import Tensor
    from torch._ops import _OpNamespace

OptimizerName: TypeAlias = Literal["saga", "newton"]

_lock = threading.Lock()
_loaded = {}


def _nvrtc(torch: ModuleType) -> ctypes.CDLL:
    """Find the installed CUDA runtime compiler without requiring an nvcc executable.

    ``torch`` supplies the CUDA major version and installation root. Search
    matching NVIDIA package/toolkit libraries before the system loader, preferring
    short shared-library names. Return a ctypes CDLL owner; raise ImportError if
    no NVRTC library can be located. No device kernel is launched here.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    base = Path(torch.__file__).parent.parent / "nvidia"
    roots = [base / f"cu{torch.version.cuda.split('.')[0]}" / "lib", base / "cuda_nvrtc" / "lib"]
    if CUDA_HOME:
        roots.extend([Path(CUDA_HOME) / "lib64", Path(CUDA_HOME) / "lib"])
    for root in roots:
        matches = sorted(root.glob("libnvrtc.so*"), key=lambda p: len(p.name))
        if matches:
            return ctypes.CDLL(str(matches[0]))
    path = ctypes.util.find_library("nvrtc")
    if path:
        return ctypes.CDLL(path)
    raise ImportError(
        "GPU classifier optimizers require the CUDA NVRTC runtime library (nvidia-cuda-nvrtc or CUDA toolkit)."
    )


def _compile(
    torch: ModuleType,
    source: bytes,
    options: Sequence[bytes],
    target: Path,
    algorithm: OptimizerName,
) -> None:
    """Compile source bytes to an atomically published PTX cache file.

    Parameters
    ----------
    torch : module
        Installed Torch used to locate a compatible NVRTC shared library.
    source : bytes
        Standalone CUDA source with no external header inputs.
    options : list of bytes
        Architecture, language, precision, and arithmetic compiler flags.
    target : pathlib.Path
        Final PTX path; the caller must hold its process-safe build lock.
    algorithm : str
        Optimizer name used for source and compiler diagnostics.

    Notes
    -----
    Write the NVRTC log beside the target, build through a temporary file, and
    replace the final path only after successful compilation. The NVRTC program
    is destroyed even on exceptions. Compilation errors raise RuntimeError with
    the compiler log; no CUDA launch or caller RNG operation occurs.
    """
    nv = _nvrtc(torch)
    program = ctypes.c_void_p()
    status = nv.nvrtcCreateProgram(
        ctypes.byref(program), source, f"ltm_{algorithm}.cu".encode(), 0, None, None
    )
    if status:
        raise RuntimeError(f"NVRTC program creation failed with code {status}")
    try:
        arguments = (ctypes.c_char_p * len(options))(*options)
        status = nv.nvrtcCompileProgram(program, len(options), arguments)
        length = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(program, ctypes.byref(length))
        log = ctypes.create_string_buffer(length.value)
        nv.nvrtcGetProgramLog(program, log)
        target.with_suffix(".log").write_bytes(log.raw)
        if status:
            raise RuntimeError(
                f"{algorithm} device compilation failed: {log.value.decode(errors='replace')}"
            )
        nv.nvrtcGetPTXSize(program, ctypes.byref(length))
        payload = ctypes.create_string_buffer(length.value)
        nv.nvrtcGetPTX(program, payload)
        temporary = target.with_suffix(".building")
        temporary.write_bytes(payload.raw)
        temporary.replace(target)
    finally:
        nv.nvrtcDestroyProgram(ctypes.byref(program))


def operators(prototype: Tensor, *, algorithm: OptimizerName = "saga") -> tuple[_OpNamespace, str]:
    """Return registered host operators and a device-compatible cached PTX path.

    Parameters
    ----------
    prototype : Tensor
        CUDA FP32/FP64 tensor supplying device and precision only; data is unread.
    algorithm : {"saga", "newton"}
        Compiled optimizer whose standalone kernels and setup operator are loaded.

    Returns
    -------
    tuple of namespace and str
        ``torch.ops.tabgen_eval_cuda`` and its prepared optimizer PTX filename. The host
        setup retains all required kernel handles on the prototype's device.

    Raises
    ------
    ValueError
        The optimizer name is unsupported.
    TypeError
        Prototype is not a CUDA floating tensor in a supported precision.
    ImportError
        Host optimizer operators or the NVRTC runtime library are unavailable.
    RuntimeError
        Device compilation or module setup fails.

    Notes
    -----
    In-process entries are keyed by optimizer, device, dtype and compute capability.
    Disk identities include source bytes, CUDA version, and exact NVRTC options;
    ``TABGEN_EVAL_CUDA_CACHE`` overrides Torch's build-cache root. File locking
    allows concurrent processes to reuse a completed PTX artifact. The current
    CUDA device is restored after setup; prototype storage and RNGs are unchanged.
    """
    import fcntl

    import torch
    from torch.utils.cpp_extension import get_default_build_root

    if algorithm not in ("saga", "newton"):
        raise ValueError(f"Unknown CUDA optimizer {algorithm!r}")
    if prototype.dtype not in (torch.float32, torch.float64) or not prototype.is_cuda:
        raise TypeError("CUDA optimizers require CUDA float32 or float64 inputs")
    device = prototype.device
    capability = torch.cuda.get_device_capability(device)
    key = (algorithm, device.index, prototype.dtype, capability)
    with _lock:
        if key not in _loaded:
            ops = library()
            if not hasattr(ops, f"{algorithm}_fit"):
                raise ImportError(
                    f"The selected prebuilt CUDA library lacks {algorithm}; rebuild it for this package version."
                )
            source = device_source(algorithm).read_bytes()
            options = [
                f"--gpu-architecture=compute_{capability[0]}{capability[1]}".encode(),
                b"--std=c++17",
                b"--fmad=false",
            ]
            if prototype.dtype == torch.float32:
                options.append(f"-DLTM_{algorithm.upper()}_FLOAT=1".encode())
            identity = hashlib.sha256(
                source + repr((torch.version.cuda, options)).encode()
            ).hexdigest()[:20]
            cache = (
                Path(os.environ.get("TABGEN_EVAL_CUDA_CACHE", get_default_build_root()))
                / f"tabgen_eval_{algorithm}"
                / identity
            )
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / "kernels.ptx"
            with (cache / "build.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not target.exists():
                    _compile(torch, source, options, target, algorithm)
            with torch.cuda.device(device):
                getattr(ops, f"{algorithm}_setup")(str(target))
            _loaded[key] = (ops, str(target))
        return _loaded[key]
