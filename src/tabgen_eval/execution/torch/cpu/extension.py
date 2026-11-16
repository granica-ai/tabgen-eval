"""Load or build Torch-ABI-specific compiled ATen CPU operators lazily.

An explicit ``TABGEN_EVAL_CPU_LIBRARY`` path takes priority over a compatible wheel
artifact, followed by a source build. A process lock protects registration; a
filesystem lock and atomic library rename protect the shared source-build cache.
Cache identity includes Torch ABI, compiler/flags, native dependencies, loader
sources, and host hardware for nonportable builds. Importing this module alone
does not import Torch or compile anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from tabgen_eval.execution.torch.binary import discover, release_rpaths
from tabgen_eval.execution.torch.sources import inventory, source_digest

if TYPE_CHECKING:
    from torch._ops import _OpNamespace


_lock = threading.Lock()
_loaded = False


_TOOLCHAIN_HELP = (
    "Install a C++17 compiler (C++20 for Torch >=2.12) and OpenMP development/runtime support "
    "(Ubuntu/Debian: sudo apt-get install build-essential; "
    "Fedora/RHEL: sudo dnf install gcc-c++ libgomp). "
    "Set CXX to a compatible compiler, for example CXX=/usr/bin/g++. "
    "Alternatively use a compatible precompiled library."
)


def _toolchain_help():
    if sys.platform == "darwin":
        return (
            "Install a C++17 compiler (C++20 for Torch >=2.12) with xcode-select --install and OpenMP with "
            "brew install libomp. Use native arm64 Python and Torch, not Rosetta. "
            "Set CXX to Apple Clang if necessary (CXX=/usr/bin/clang++). "
            "For a custom OpenMP installation, set TABGEN_EVAL_OPENMP_ROOT to its prefix."
        )
    return _TOOLCHAIN_HELP


def _supported_platform():
    return sys.platform == "linux" or (sys.platform == "darwin" and platform.machine() == "arm64")


def _mac_openmp(torch):
    """Prefer Torch's OpenMP runtime; discover headers without mixing runtime copies."""
    torch_root = Path(torch.__file__).resolve().parent
    override = os.environ.get("TABGEN_EVAL_OPENMP_ROOT")
    bundled_include = torch_root / "include"
    bundled_runtime = torch_root / "lib" / "libomp.dylib"
    if not override and (bundled_include / "omp.h").is_file() and bundled_runtime.is_file():
        return bundled_include.resolve(), bundled_runtime.resolve()
    prefixes = [Path(override).expanduser()] if override else [Path("/opt/homebrew/opt/libomp")]
    if not override and not (prefixes[0] / "include" / "omp.h").is_file():
        try:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if result.returncode == 0 and result.stdout.strip():
                prefixes.append(Path(result.stdout.strip()))
    headers = [bundled_include] if not override else []
    headers.extend(p / "include" for p in prefixes)
    runtimes = [bundled_runtime]
    runtimes.extend(p / "lib" / "libomp.dylib" for p in prefixes)
    include = next((p.resolve() for p in headers if (p / "omp.h").is_file()), None)
    library = next((p.resolve() for p in runtimes if p.is_file()), None)
    if include is None or library is None:
        raise ImportError(
            "Cannot locate macOS OpenMP headers (omp.h) and runtime (libomp.dylib). "
            + _toolchain_help()
        )
    return include, library


def _cxx_standard(torch):
    """Honor Torch's published C++20 transition and explicit installed header guards."""
    version = re.match(r"(\d+)\.(\d+)", getattr(torch, "__version__", ""))
    standard = 20 if version and tuple(map(int, version.groups())) >= (2, 12) else 17
    if getattr(torch, "__file__", None):
        include = Path(torch.__file__).resolve().parent / "include"
        for relative in ("ATen/ATen.h", "torch/csrc/api/include/torch/all.h"):
            header = include / relative
            if header.is_file() and re.search(r"__cplusplus\s*<\s*202002L?", header.read_text()):
                standard = 20
    return standard


def _build_settings(torch, *, portable=False):
    """Compiler and linker options for Linux or native Apple Silicon CPU execution."""
    flags = ["-O3", f"-std=c++{_cxx_standard(torch)}", "-fPIC", "-ffp-contract=off"]
    if sys.platform == "darwin":
        include, runtime = _mac_openmp(torch)
        openmp = ["-Xpreprocessor", "-fopenmp", f"-I{include}"]
        link = [str(runtime), f"-Wl,-rpath,{runtime.parent}", "-Wl,-headerpad_max_install_names"]
        return flags + ["-arch", "arm64"] + openmp, openmp, link, ".dylib"
    flags += ["-fopenmp"]
    if not portable:
        flags.append("-march=native")
    return flags, ["-fopenmp"], [], ".so"


def _compiler() -> list[str]:
    try:
        compiler = shlex.split(os.environ.get("CXX", "c++"))
    except ValueError as exc:
        raise ImportError(f"Invalid CXX setting: {exc}. {_toolchain_help()}") from exc
    if not compiler:
        raise ImportError(f"CXX is empty. {_toolchain_help()}")
    return compiler


def _run_build_step(command, root, stage, *, timeout, prerequisite=False):
    """Preserve diagnostics even when a compiler fails to start or times out."""
    log = root / f"{stage}.log"
    error = ImportError if prerequisite else RuntimeError
    help_text = (
        _toolchain_help()
        if prerequisite
        else (
            "Check the compiler output and that your compiler supports the installed Torch headers. "
            "A killed compiler can indicate insufficient memory. "
            "Fix the reported problem and retry; failed builds are not cached."
        )
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:

        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

        log.write_text(shlex.join(command) + "\n" + decoded(exc.stdout) + decoded(exc.stderr))
        raise error(
            f"CPU {stage} timed out after {timeout} seconds; see {log}. "
            f"Check host load and compiler wrappers. {help_text}"
        ) from exc
    except OSError as exc:
        log.write_text(shlex.join(command) + f"\n{exc}\n")
        raise error(
            f"Could not run CPU {stage}: {exc}; see {log}. {help_text} "
            "If the cache filesystem disallows execution, set TABGEN_EVAL_CPU_CACHE "
            "to a writable location that permits execution."
        ) from exc
    log.write_text(shlex.join(command) + "\n" + result.stdout + result.stderr)
    if result.returncode:
        detail = (result.stderr or result.stdout)[-3000:].strip()
        raise error(f"CPU {stage} failed; see {log}. {help_text}\n{detail}")
    return result


def _fix_macos_openmp(target, runtime, root, *, stage, prerequisite=False):
    """Repair generated binaries, never Torch's runtime, then refresh their ad-hoc signature.

    A bundled libomp may retain its builder's absolute LC_ID_DYLIB. Linking that
    file copies the stale path into our binary; an rpath cannot override it.
    Resolve every direct libomp dependency to the runtime we actually selected.
    """
    result = _run_build_step(
        ["otool", "-L", str(target)],
        root,
        f"{stage}-dependencies",
        timeout=30,
        prerequisite=prerequisite,
    )
    changed = False
    for line in result.stdout.splitlines()[1:]:
        dependency = line.strip().split(" (compatibility version", 1)[0]
        if Path(dependency).name != "libomp.dylib" or dependency == str(runtime):
            continue
        _run_build_step(
            ["install_name_tool", "-change", dependency, str(runtime), str(target)],
            root,
            f"{stage}-relocate",
            timeout=30,
            prerequisite=prerequisite,
        )
        changed = True
    if changed:
        _run_build_step(
            ["codesign", "--force", "--sign", "-", str(target)],
            root,
            f"{stage}-codesign",
            timeout=30,
            prerequisite=prerequisite,
        )


def _check_prerequisites(
    compiler,
    flags,
    root,
    *,
    openmp_flags=("-fopenmp",),
    openmp_link_flags=(),
    torch_include_paths=(),
):
    """Check the selected C++ standard, OpenMP, and actual Torch headers before the full build."""
    standard = "cxx20" if "-std=c++20" in flags else "cxx17"
    language_probe = (
        "#include <concepts>\ntemplate<class T> requires std::integral<T> "
        "int check(T x) { return x; }\nint main() { return check(0); }\n"
        if standard == "cxx20"
        else "#include <optional>\nint main() { return std::optional<int>(0).value(); }\n"
    )
    programs = {
        standard: language_probe,
        "openmp": (
            "#include <omp.h>\nint main() { int n = 0;\n"
            "#pragma omp parallel num_threads(2) reduction(+:n)\n"
            "{ n += 1; }\nreturn n > 0 ? 0 : 1; }\n"
        ),
    }
    for name, program in programs.items():
        source, target = root / f"check-{name}.cpp", root / f"check-{name}"
        source.write_text(program)
        options = flags if name == "openmp" else [f for f in flags if f not in openmp_flags]
        link = list(openmp_link_flags) if name == "openmp" else []
        _run_build_step(
            compiler + options + [str(source), "-o", str(target)] + link,
            root,
            f"check-{name}",
            timeout=60,
            prerequisite=True,
        )
        if name == "openmp" and sys.platform == "darwin" and openmp_link_flags:
            _fix_macos_openmp(
                target,
                Path(openmp_link_flags[0]),
                root,
                stage="check-openmp",
                prerequisite=True,
            )
        _run_build_step([str(target)], root, f"check-{name}-runtime", timeout=30, prerequisite=True)
        target.unlink()
    if torch_include_paths:
        source, target = root / "check-torch-headers.cpp", root / "check-torch-headers.o"
        source.write_text("#include <ATen/ATen.h>\n#include <torch/library.h>\n")
        _run_build_step(
            compiler
            + flags
            + [f"-I{p}" for p in torch_include_paths]
            + ["-c", str(source), "-o", str(target)],
            root,
            "check-torch-headers",
            timeout=120,
            prerequisite=True,
        )
        target.unlink()


def library() -> _OpNamespace:
    """Load the ATen CPU operators once per process.

    A compatible bundled library takes precedence over a source build. The
    explicit ``TABGEN_EVAL_CPU_LIBRARY`` override takes precedence over both.

    Returns
    -------
    namespace
        Registered operators under ``torch.ops.tabgen_eval_cpu``.

    Raises
    ------
    ImportError
        No compatible library or supported source-build toolchain is available.
    RuntimeError
        Source compilation fails; the exception includes the build-log path.
    """
    global _loaded
    import torch

    if _loaded:
        return torch.ops.tabgen_eval_cpu
    with _lock:
        if _loaded:
            return torch.ops.tabgen_eval_cpu
        prebuilt = os.environ.get("TABGEN_EVAL_CPU_LIBRARY")
        if prebuilt:
            torch.ops.load_library(prebuilt)
        else:
            packaged, diagnostic = discover(torch, "cpu")
            if packaged:
                torch.ops.load_library(str(packaged))
            else:
                try:
                    _build(torch)
                except (ImportError, RuntimeError) as exc:
                    if diagnostic:
                        raise ImportError(f"{diagnostic} Source fallback failed: {exc}") from exc
                    raise
        _loaded = True
    return torch.ops.tabgen_eval_cpu


def available() -> bool:
    """Check for a bundled library, explicit override, or source compiler.

    Returns
    -------
    bool
        Whether execution has a potential loading or compilation path. This
        check does not compile code or initialize a CUDA device.
    """
    import shutil

    import torch

    if _loaded:
        return True
    override = os.environ.get("TABGEN_EVAL_CPU_LIBRARY")
    if override:
        return Path(override).is_file()
    if discover(torch, "cpu")[0]:
        return True
    try:
        compiler = _compiler()
    except ImportError:
        return False
    return _supported_platform() and bool(compiler and shutil.which(compiler[0]))


def _build(torch: ModuleType, *, portable: bool = False, load: bool = True) -> Path:
    """Build or reuse CPU operators against the installed Torch ABI.

    Parameters
    ----------
    torch : module
        Installed PyTorch supplying C++ headers, libraries, and ABI settings.
    portable : bool, default=False
        Use wheel flags and relative runtime search paths; omit host-specific
        instruction selection and hardware from cache identity.
    load : bool, default=True
        Register the completed library. Packaging callers disable registration.

    Returns
    -------
    pathlib.Path
        Completed shared library beneath ``TABGEN_EVAL_CPU_CACHE`` or Torch's default
        extension cache, keyed by the full source/toolchain identity.

    Notes
    -----
    Linux and Apple Silicon macOS builds use a filesystem lock, compile to a temporary library, and
    publish by atomic rename. Cache hits skip compilation. ``CXX`` may include
    compiler arguments. Compiler output and build metadata are retained beside the
    artifact; a failed compilation leaves no reusable target library.

    Raises
    ------
    ImportError
        The platform, compiler, C++17 standard library or OpenMP is unavailable.
    RuntimeError
        Compilation exits unsuccessfully; the error points to ``build.log``.
    """
    if not _supported_platform():
        raise ImportError(
            "CPU source builds support Linux and Apple Silicon macOS (arm64). "
            "On Apple Silicon, install native arm64 Python and Torch instead of Rosetta/x86_64. "
            "Windows and Intel macOS source builds are not supported."
        )
    import fcntl

    from torch.utils.cpp_extension import get_default_build_root, include_paths, library_paths

    native = inventory("cpu")
    sources = native.compilation
    compiler = _compiler()
    flags, openmp, openmp_link, suffix = _build_settings(torch, portable=portable)
    hardware = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    hardware = "\n".join(
        line for line in hardware.splitlines() if line.startswith(("model name", "flags"))
    )
    identity = hashlib.sha256()
    identity.update(
        repr(
            (
                torch.__version__,
                torch.compiled_with_cxx11_abi(),
                compiler,
                flags,
                openmp_link,
                sys.platform,
                platform.machine(),
                hardware if not portable else None,
            )
        ).encode()
    )
    identity.update(Path(__file__).read_bytes())
    identity.update((Path(__file__).parent.parent / "binary.py").read_bytes())
    identity.update(source_digest("cpu").encode())
    root = (
        Path(os.environ.get("TABGEN_EVAL_CPU_CACHE", get_default_build_root()))
        / "tabgen_eval_cpu"
        / identity.hexdigest()[:20]
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock_file = (root / "build.lock").open("w")
    except OSError as exc:
        raise ImportError(
            f"Cannot write the CPU build cache at {root}: {exc}. "
            "Set TABGEN_EVAL_CPU_CACHE to a writable directory with enough free space."
        ) from exc
    target = root / f"tabgen_eval_cpu{suffix}"
    with lock_file as lock:
        # Separate Python processes may reach the same cache entry together.
        # Publish only a completed shared library while holding this file lock.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            torch_includes = include_paths()
            _check_prerequisites(
                compiler,
                flags,
                root,
                openmp_flags=openmp,
                openmp_link_flags=openmp_link,
                torch_include_paths=torch_includes,
            )
            command = compiler + flags
            if sys.platform == "linux":
                command += [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]
            command += [f"-I{native.root}"] + [f"-I{path}" for path in torch_includes]
            command += [str(source) for source in sources]
            command += [
                "-dynamiclib" if sys.platform == "darwin" else "-shared",
                "-o",
                str(root / f"building{suffix}"),
            ]
            command += [f"-L{path}" for path in library_paths()]
            command += [
                f"-Wl,-rpath,{path}"
                for path in (
                    release_rpaths() if portable and sys.platform == "linux" else library_paths()
                )
                if f"-Wl,-rpath,{path}" not in openmp_link
            ]
            command += ["-ltorch_cpu", "-lc10"] + openmp_link
            if sys.platform == "linux":
                command += ["-ldl"]
            _run_build_step(command, root, "build", timeout=600)
            if sys.platform == "darwin":
                _fix_macos_openmp(
                    root / f"building{suffix}",
                    Path(openmp_link[0]),
                    root,
                    stage="build",
                )
            (root / f"building{suffix}").replace(target)
            (root / "build.json").write_text(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "command": command,
                        "sources": [str(p) for p in sources],
                    },
                    indent=2,
                )
            )
    if load:
        torch.ops.load_library(str(target))
    return target
