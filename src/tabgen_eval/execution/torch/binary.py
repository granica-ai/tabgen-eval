"""Discover and validate Torch-specific operator libraries bundled in wheels.

This module owns artifact manifests, ELF runtime requirements, checksums, and
loading support. It imports no Torch itself: callers supply their installed
module. Discovery returns compatibility diagnostics without compiling or loading;
CPU/CUDA extension loaders choose whether to fall back to source compilation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Literal

from tabgen_eval.execution.torch.sources import source_digest as native_source_digest

ROOT = Path(__file__).parents[1]
SCHEMA = 1
_dependencies = []


def source_digest(kind: Literal["cpu", "cuda"]) -> str:
    """Return the native dependency fingerprint for the requested backend.

    Delegate to the shared inventory using this package's native tree, so wheel
    compatibility and source builds agree about header/device ownership. Unknown
    backends raise ``ValueError``; reading source bytes does not load Torch.
    """
    return native_source_digest(kind, root=ROOT / "csrc")


def identity(torch: ModuleType, kind: Literal["cpu", "cuda"]) -> dict[str, object]:
    """Describe the installed Torch ABI and native sources required by an artifact.

    Return JSON-compatible schema, backend, Torch version, C++ ABI, platform,
    architecture, CUDA build version (CUDA only), and source digest. This describes
    compatibility, not whether a GPU or compiler is available, and does not load the
    library or initialize a device. ``kind`` follows the native inventory contract.
    """
    return {
        "schema": SCHEMA,
        "kind": kind,
        "torch": str(torch.__version__),
        "cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        "system": platform.system(),
        "machine": platform.machine(),
        "cuda": torch.version.cuda if kind == "cuda" else None,
        "sources_sha256": source_digest(kind),
    }


def manifest(torch: ModuleType, kind: Literal["cpu", "cuda"], library: Path) -> dict[str, object]:
    """Record compatibility, checksum and actual ELF runtime requirements.

    Parameters
    ----------
    torch : module
        PyTorch used to compile the library.
    kind : {"cpu", "cuda"}
        The operator family and corresponding source package.
    library : pathlib.Path
        A completed shared library inspected with ``readelf`` during packaging.

    Returns
    -------
    dict
        JSON-serializable compatibility metadata. GLIBCXX requirements are
        recorded; their enforcement is performed by the system dynamic linker.
    """
    result = identity(torch, kind)
    result["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    result["cpu_target"] = "compiler_baseline"
    # The version requirements come from the artifact, rather than the build host.
    output = subprocess.run(
        ["readelf", "--version-info", str(library)], capture_output=True, text=True, check=True
    ).stdout
    versions = set(re.findall(r"\bGLIBC_([0-9.]+)\b", output))
    result["glibc_min"] = (
        max(versions, key=lambda v: tuple(map(int, v.split(".")))) if versions else None
    )
    result["glibcxx_required"] = sorted(set(re.findall(r"\bGLIBCXX_([0-9.]+)\b", output)))
    return result


def discover(torch: ModuleType, kind: Literal["cpu", "cuda"]) -> tuple[Path | None, str | None]:
    """Find a bundled library and explain compatibility rejection without compiling.

    ``torch`` is the installed module; ``kind`` selects ``cpu`` or ``cuda`` under
    ``execution/_binaries``. Compare the manifest with the full installed identity,
    library checksum, and minimum glibc. GLIBCXX requirements are recorded by
    ``manifest`` but left to the system loader to enforce.

    Return ``(Path, None)`` on success, ``(None, diagnostic)`` on rejection, or
    ``(None, None)`` when no artifact or metadata exists. Malformed/unreadable
    metadata is reported as a diagnostic. This function does not load operators;
    it does read and hash packaged files.
    """
    directory = ROOT / "_binaries" / kind
    path = directory / f"tabgen_eval_{kind}.so"
    metadata = directory / "metadata.json"
    if not path.exists() and not metadata.exists():
        return None, None
    try:
        data = json.loads(metadata.read_text())
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        for key, value in identity(torch, kind).items():
            if data.get(key) != value:
                return (
                    None,
                    f"Bundled {kind} library requires {key}={data.get(key)!r}; installed value is {value!r}.",
                )
        expected = data.get("library_sha256")
        if expected != hashlib.sha256(path.read_bytes()).hexdigest():
            return None, f"Bundled {kind} library checksum does not match metadata."
        required = data.get("glibc_min")
        libc, version = platform.libc_ver()
        if required and (
            libc != "glibc"
            or tuple(map(int, version.split("."))) < tuple(map(int, required.split(".")))
        ):
            return (
                None,
                f"Bundled {kind} library requires glibc >= {required}; installed runtime is {libc} {version}.",
            )
        return path, None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"Bundled {kind} library metadata is invalid: {exc}."


def release_rpaths() -> list[str]:
    """Return relocatable ELF library search paths for bundled operator libraries.

    Paths are relative to ``execution/_binaries/{cpu,cuda}`` and cover Torch and
    supported NVIDIA dependency layouts under site-packages. The literal ELF
    ``$ORIGIN`` token is preserved for the linker/loader, not expanded by Python.
    """
    base = "$ORIGIN/../../../.."
    return [
        f"{base}/torch/lib",
        *[
            f"{base}/nvidia/{name}/lib"
            for name in (
                "cu12",
                "cu13",
                "cuda_runtime",
                "cublas",
                "cusolver",
                "cusparse",
                "nvjitlink",
                "cufft",
                "curand",
            )
        ],
    ]


def load(torch: ModuleType, kind: Literal["cpu", "cuda"], path: Path) -> None:
    """Register a discovered native library with optional CUDA dependency preloading.

    ``path`` must already have passed compatibility checks. For CUDA, load Torch's
    split linalg library globally when present so symbols also resolve for a
    separate ``--target`` installation. Retain its ctypes handle for process lifetime.
    Operator registration and loader errors propagate from Torch/system loading;
    this function neither compiles sources nor returns an operator namespace.
    """
    if kind == "cuda":
        linalg = Path(torch.__file__).parent / "lib" / "libtorch_cuda_linalg.so"
        if linalg.is_file():
            _dependencies.append(ctypes.CDLL(str(linalg), mode=ctypes.RTLD_GLOBAL))
    torch.ops.load_library(str(path))
