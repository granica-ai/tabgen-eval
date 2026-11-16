"""Lazily load Torch-ABI-specific CUDA host operators.

Prefer an explicit library override, then a compatible bundled binary, then a
source build using the central native inventory. Operators cover native dense
linear algebra, Newton/SAGA training, SAGA preparation, and optional Triton TV
orchestration; optimizer device PTX has its separate lazy NVRTC cache. A thread
lock publishes one registered operator namespace per process, and source-build file locking prevents
partially compiled libraries from being reused across processes.

Importing this module does not import Torch, inspect a GPU, or compile code.
Torch and toolchain inspection occur at availability/loading entry points.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from tabgen_eval.execution.torch.binary import discover, load, release_rpaths
from tabgen_eval.execution.torch.sources import inventory, source_digest

if TYPE_CHECKING:
    from torch._ops import _OpNamespace


_loaded = False
_lock = threading.Lock()


def library() -> _OpNamespace:
    """Load and register the CUDA host operators once in this process.

    An explicit ``TABGEN_EVAL_CUDA_LIBRARY`` override precedes compatible bundled
    binaries and source compilation. Binary compatibility/checksums belong to
    ``execution.torch.binary``; native source ownership belongs to its central inventory.

    Returns
    -------
    namespace
        Registered ``torch.ops.tabgen_eval_cuda`` operators, with supported fake
        implementations installed after the real library is registered.

    Raises
    ------
    ImportError
        A compatible library or supported source toolchain cannot be loaded.
    RuntimeError
        Host compilation fails; the exception includes its build-log path.

    Notes
    -----
    The thread lock protects first-use registration; the loaded flag is published
    only after library and fake implementations have registered successfully.
    Success is retained for process lifetime; later environment override changes
    have no effect. This loads the host library, not shape-specialized Triton
    kernels or optimizer PTX, and does not allocate model tensors or change RNG state.
    """
    global _loaded
    import torch

    if not _loaded:
        with _lock:
            if not _loaded:
                path = os.environ.get("TABGEN_EVAL_CUDA_LIBRARY")
                if path:
                    load(torch, "cuda", path)
                else:
                    packaged, diagnostic = discover(torch, "cuda")
                    if packaged:
                        load(torch, "cuda", packaged)
                    else:
                        try:
                            _build(torch)
                        except (ImportError, RuntimeError) as exc:
                            if diagnostic:
                                raise ImportError(
                                    f"{diagnostic} Source fallback failed: {exc}"
                                ) from exc
                            raise
                _register_fake(torch)
                _loaded = True
    return torch.ops.tabgen_eval_cuda


def _register_fake(torch: ModuleType) -> None:
    """Register shape/layout-only implementations for supported CUDA host operators.

    ``torch`` is the already loaded runtime and real schemas must be registered.
    Fake outputs retain the represented dtype/device and column-major matrix
    layouts without allocating CUDA data or running cuBLAS/cuSOLVER. Optional
    GEMM/TV registrations tolerate older host libraries; the optimizers' dynamic state
    operators are not assigned fake implementations here. Called once under the
    library lock because duplicate registrations are not idempotent.
    """
    if hasattr(torch.ops.tabgen_eval_cuda, "dense_tv_batches"):
        torch.library.register_fake("tabgen_eval_cuda::dense_tv_batches")(
            lambda a, b, pairs, levels, prototype, grid, batch, cond, histogram, reduction: (
                [prototype.new_empty((a.shape[0], pairs.shape[0]))]
                if not a.shape[0] or not pairs.shape[0]
                else [
                    prototype.new_empty((a.shape[0], min(batch, pairs.shape[0] - start)))
                    for start in range(0, pairs.shape[0], batch)
                ]
            )
        )
    torch.library.register_fake("tabgen_eval_cuda::mv")(
        lambda x, v, transpose=False: v.new_empty(x.shape[1] if transpose else x.shape[0])
    )
    if hasattr(torch.ops.tabgen_eval_cuda, "gemm"):
        torch.library.register_fake("tabgen_eval_cuda::gemm")(
            lambda a, b: a.new_empty((b.shape[1], a.shape[0])).T
        )
    torch.library.register_fake("tabgen_eval_cuda::lower_gram")(
        lambda x: x.new_empty((x.shape[1], x.shape[1])).T
    )
    torch.library.register_fake("tabgen_eval_cuda::cholesky")(
        lambda x: (x.new_empty(x.shape).T, x.new_empty((1,), dtype=torch.int32))
    )
    torch.library.register_fake("tabgen_eval_cuda::solve")(
        lambda factor, rhs: (rhs.new_empty(rhs.shape), rhs.new_empty((1,), dtype=torch.int32))
    )

    torch.library.register_fake("tabgen_eval_cuda::potrf_workspace")(
        lambda x: 8 * x.numel() * x.element_size()
    )


def available() -> bool:
    """Check whether host-library loading or source compilation is plausible.

    Return True for an already loaded namespace, an existing explicit override,
    a compatible bundled library, or Linux CUDA Torch with required developer
    headers/libraries and a discoverable C++ compiler. This does not load an
    explicit binary to verify its ABI, compile code, require a free GPU, or prove
    NVRTC/Triton device kernels can run. Source discovery failures return False.
    """
    if _loaded:
        return True
    if os.environ.get("TABGEN_EVAL_CUDA_LIBRARY"):
        return Path(os.environ["TABGEN_EVAL_CUDA_LIBRARY"]).is_file()
    import torch

    if sys.platform != "linux" or not torch.version.cuda:
        return False
    packaged, _ = discover(torch, "cuda")
    if packaged:
        return True
    try:
        _sdk(torch)
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        return bool(compiler and shutil.which(compiler[0]))
    except (ImportError, OSError):
        return False


def _sdk(torch: ModuleType) -> tuple[list[Path], list[Path], list[Path]]:
    """Locate CUDA headers and link libraries matching the installed Torch runtime.

    Parameters
    ----------
    torch : module
        Loaded CUDA Torch runtime used to choose the NVIDIA package major version.

    Returns
    -------
    tuple of list
        Header directories, library directories, and selected cuBLAS/cuSOLVER/
        cudart shared-library paths. Search installed NVIDIA layouts and CUDA_HOME,
        then append user header roots from ``TABGEN_EVAL_CUDA_INCLUDE``.

    Raises
    ------
    ImportError
        Runtime/BLAS/solver, developer CRT, CCCL headers, or required libraries
        cannot be found. Discovery does not compile or load a device module.

    Notes
    -----
    Support both merged ``nvidia/cuXX`` and split wheel/toolkit layouts. Header
    and library existence is checked here; full compiler/linker compatibility is
    ultimately established by the native build, not inferred from filenames alone.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    base = Path(torch.__file__).parent.parent / "nvidia"
    major = int(torch.version.cuda.split(".")[0])
    candidates = [base / f"cu{major}"]
    # Torch wheel libraries must precede a system toolkit. An unversioned
    # system libcublas.so must not win over Torch's versioned libcublas.so.11:
    # the native operators consume cuBLAS handles owned by Torch.
    candidates.extend(
        base / name
        for name in ("cuda_runtime", "cuda_nvcc", "cuda_nvrtc", "cublas", "cusolver", "cusparse")
    )
    if CUDA_HOME:
        toolkit = Path(CUDA_HOME)
        header = toolkit / "include" / "cuda.h"
        match = (
            re.search(r"#define\s+CUDA_VERSION\s+(\d+)", header.read_text())
            if header.is_file()
            else None
        )
        if match and int(match[1]) // 1000 == major:
            candidates.append(toolkit)
    includes, libs = [], []
    for root in candidates:
        if (root / "include").is_dir():
            includes.append(root / "include")
        for name in ("lib", "lib64"):
            if (root / name).is_dir():
                libs.append(root / name)
    includes.extend(
        Path(p) for p in os.environ.get("TABGEN_EVAL_CUDA_INCLUDE", "").split(os.pathsep) if p
    )
    includes.extend(p / "cccl" for p in list(includes) if (p / "cccl").is_dir())
    for header in ("cuda_runtime.h", "cublas_v2.h", "cusolverDn.h"):
        if not any((p / header).exists() for p in includes):
            raise ImportError(
                f"GPU Linear C2ST requires CUDA headers ({header}) or TABGEN_EVAL_CUDA_LIBRARY."
            )
    if not any((p / "crt" / "host_defines.h").exists() for p in includes):
        raise ImportError(
            "GPU Linear C2ST requires matching CUDA developer/CRT headers; use CUDA_HOME, TABGEN_EVAL_CUDA_INCLUDE, or a compatible prebuilt library."
        )
    if major >= 12 and not any((p / "nv" / "target").exists() for p in includes):
        raise ImportError(
            "GPU Linear C2ST requires matching CUDA CCCL developer headers; use CUDA_HOME or TABGEN_EVAL_CUDA_INCLUDE."
        )
    libraries = []
    for name in ("cublas", "cusolver", "cudart"):
        matches = []
        for root in libs:
            matches = [p for p in root.glob(f"lib{name}.so*") if p.is_file()]
            if matches:
                break
        if not matches:
            raise ImportError(f"GPU Linear C2ST requires lib{name} or TABGEN_EVAL_CUDA_LIBRARY.")
        libraries.append(sorted(matches, key=lambda p: len(p.name))[0])
    return includes, libs, libraries


def _build(torch: ModuleType, *, portable: bool = False, load: bool = True) -> Path:
    """Build or reuse host operators against the installed Torch/CUDA ABI.

    Parameters
    ----------
    torch : module
        Torch runtime supplying headers, link paths, and C++ ABI identity.
    portable : bool, default=False
        Use wheel-relative runtime search paths rather than environment paths.
    load : bool, default=True
        Register the completed shared library; wheel builders disable loading.

    Returns
    -------
    pathlib.Path
        Completed ``tabgen_eval_cuda.so`` in a source/toolchain-keyed build directory.

    Raises
    ------
    ImportError
        The platform, SDK, or C++ compiler cannot support source compilation.
    RuntimeError
        Compilation/linking fails; details are written to the cache's build.log.

    Notes
    -----
    Hash compiler/arithmetic flags, Torch ABI/version, SDK paths, loader/binary
    code, and recursively inventoried native dependencies. A per-directory file
    lock serializes builders; successful output replaces a temporary library
    atomically. ``TABGEN_EVAL_CUDA_CACHE`` selects the cache root. Compile with
    FP contraction disabled; this host build does not invoke nvcc or NVRTC, create
    model tensors, or depend on a particular GPU architecture.
    """
    if sys.platform != "linux" or not torch.version.cuda:
        raise ImportError(
            "GPU Linear C2ST source builds require Linux CUDA PyTorch; use a compatible TABGEN_EVAL_CUDA_LIBRARY elsewhere."
        )
    import fcntl

    from torch.utils.cpp_extension import get_default_build_root, include_paths, library_paths

    includes, libs, libraries = _sdk(torch)
    native = inventory("cuda")
    sources = native.compilation
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-O3", "-std=c++17", "-fPIC", "-ffp-contract=off"]
    identity = hashlib.sha256(
        repr(
            (
                torch.__version__,
                torch.compiled_with_cxx11_abi(),
                compiler,
                flags,
                portable,
                includes,
                libraries,
            )
        ).encode()
    )
    identity.update(Path(__file__).read_bytes())
    identity.update((Path(__file__).parent.parent / "binary.py").read_bytes())
    identity.update(source_digest("cuda").encode())
    root = (
        Path(os.environ.get("TABGEN_EVAL_CUDA_CACHE", get_default_build_root()))
        / "tabgen_eval_cuda"
        / identity.hexdigest()[:20]
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / "tabgen_eval_cuda.so"
    with (root / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            paths = library_paths() + [str(p) for p in libs]
            command = (
                compiler
                + flags
                + [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]
            )
            command += [f"-I{native.root}"]
            command += [f"-I{p}" for p in include_paths() + [str(p) for p in includes]]
            command += [str(s) for s in sources] + ["-shared", "-o", str(root / "building.so")]
            command += [f"-L{p}" for p in paths] + [
                f"-Wl,-rpath,{p}" for p in (release_rpaths() if portable else paths)
            ]
            command += [
                "-ltorch_cuda",
                "-ltorch_cuda_linalg",
                "-lc10_cuda",
                "-ltorch_cpu",
                "-lc10",
                "-l:libcuda.so.1",
            ] + [str(p) for p in libraries]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            except FileNotFoundError as exc:
                raise ImportError(
                    "GPU Linear C2ST requires a C++ compiler or a compatible prebuilt CUDA library."
                ) from exc
            (root / "build.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"CUDA host operator compilation failed; see {root / 'build.log'}\n{result.stderr[-3000:]}"
                )
            (root / "building.so").replace(target)
    if load:
        torch.ops.load_library(str(target))
    return target
