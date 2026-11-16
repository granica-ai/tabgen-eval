"""Distribute portable C launchers for the finite Triton host signatures.

Wheel generation enumerates our kernel argument layouts, compiles Triton's C
CUDA driver/launcher modules, and records ABI/source/checksum/GLIBC metadata.
Installation validates the complete bundle before populating Triton's configured
cache, leaving its cache-manager and compiler hooks intact. Device kernels still
compile normally with Triton's LLVM/PTX tools for each requested specialization.

This module does not import Triton or initialize CUDA at import time. Kernel
modules call ``install`` when imported; source checkouts without a bundle keep
Triton's normal host compilation path. Bundle source identities include kernel
source bytes, so even documentation changes invalidate a generated bundle.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
import json
import platform
import re
import subprocess
import sysconfig
import threading
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).parents[3]
_installed_cache = None
_lock = threading.Lock()
_SCALARS = {
    "N",
    "NQ",
    "NR",
    "NS",
    "STRIDE",
    "P",
    "Q",
    "COND",
    "START",
    "QUERY_START",
    "REFERENCE_START",
}
_INLINE = {"_search", "_distances", "_divide", "_sign", "_finite"}
_MODULES = (
    "concentration",
    "distance",
    "fidelity",
    "fidelity_grouped",
    "classifier",
    "preprocessing",
)


def identity() -> dict[str, object]:
    """Describe the exact Python/Triton host ABI and kernel source identity.

    Return a fresh mapping with schema, Triton version, Python SOABI, system,
    machine, and a SHA-256 digest of Triton's driver/build code, this module, and
    the owned kernel modules. Import Triton lazily without loading the native
    metric extension. Exact source hashing intentionally makes incompatible
    bundles fail closed, including after source-only documentation edits.
    """
    import triton
    from triton.backends.nvidia import driver
    from triton.runtime import build

    digest = hashlib.sha256()
    for path in (
        Path(driver.__file__),
        Path(driver.__file__).with_name("driver.c"),
        Path(build.__file__),
        Path(__file__),
        *[ROOT / "torch" / "cuda" / "triton" / f"{module}.py" for module in _MODULES],
    ):
        digest.update(path.name.encode() + path.read_bytes())
    return {
        "schema": 1,
        "triton": triton.__version__,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "system": platform.system(),
        "machine": platform.machine(),
        "triton_sources_sha256": digest.hexdigest(),
    }


def launcher_sources() -> tuple[list[str], dict[str, dict[str, object]]]:
    """Enumerate host launch layouts independently of dataset dimensions.

    Returns
    -------
    sources : list of str
        Deduplicated C launcher sources covering pointer layouts and all Python
        integer widths, including Triton's special constexpr treatment of one.
    inventory : dict
        Argument positions for every launched kernel in ``torch.cuda.triton``.

    Notes
    -----
    Pointer element types and constexpr values do not change host C layouts.
    Triton 3.2 omits declared constexpr arguments from launcher signatures and
    carries specialized scalars in a separate constants map. Newer launchers
    represent both with explicit constexpr signature entries.
    Inline-only search, distance, division and finite-value helpers are excluded.
    """
    import triton
    from triton.backends.nvidia.driver import make_launcher

    legacy_signature = "ids" in inspect.signature(make_launcher).parameters
    sources = set()
    inventory = {}
    for module_name in _MODULES:
        module = importlib.import_module(f"tabgen_eval.execution.torch.cuda.triton.{module_name}")
        for name, fn in vars(module).items():
            if not isinstance(fn, triton.runtime.JITFunction) or name in _INLINE:
                continue
            scalar_positions = [
                p.num for p in fn.params if not p.is_constexpr and p.name in _SCALARS
            ]
            inventory[f"{module_name}.{name}"] = {
                "constexpr": [p.num for p in fn.params if p.is_constexpr],
                "integers": scalar_positions,
                "arguments": len(fn.params),
            }
            declared_constants = [p.num for p in fn.params if p.is_constexpr]
            for widths in itertools.product(
                ("i32", "u32", "i64", "u64", None), repeat=len(scalar_positions)
            ):
                constants = dict.fromkeys(declared_constants, 1)
                signature = {p.num: "*fp32" for p in fn.params if not p.is_constexpr}
                for position, kind in zip(scalar_positions, widths, strict=True):
                    signature[position] = "i32" if kind is None else kind
                    if kind is None:
                        constants[position] = 1
                if legacy_signature:
                    sources.add(
                        make_launcher(
                            constants, signature, {"ids_of_const_exprs": declared_constants}
                        )
                    )
                else:
                    signature = {
                        p.num: "constexpr" if p.num in constants else signature[p.num]
                        for p in fn.params
                    }
                    sources.add(make_launcher({}, signature, None))
    return sorted(sources), inventory


def build_bundles(directory: Path) -> dict[str, object]:
    """Build portable Triton driver and launcher modules for a binary wheel.

    Parameters
    ----------
    directory : pathlib.Path
        Destination for source-keyed shared libraries and compatibility metadata.
        An existing compatible library may be reused by its exact source key.

    Returns
    -------
    dict
        Bundle metadata including Python/Triton identity, kernel inventory,
        checksums and the actual minimum GLIBC version.

    Notes
    -----
    The build uses a C compiler, Python headers, Triton's bundled CUDA headers
    and the installed CUDA driver library. It does not initialize a CUDA device
    or compile any shape-specific device kernels.
    """
    from triton.backends.nvidia import driver
    from triton.runtime import build

    metadata = identity()
    if not metadata["triton"].startswith("3.6."):
        raise RuntimeError("Portable Triton host wrappers currently target Triton 3.6")
    directory.mkdir(parents=True, exist_ok=True)
    sources, inventory = launcher_sources()
    modules = [("cuda_utils", Path(driver.__file__).with_name("driver.c").read_text())]
    modules += [("__triton_launcher", source) for source in sources]
    entries = []
    glibc_versions = set()
    for name, source in modules:
        key = hashlib.sha256((source + build.platform_key()).encode()).hexdigest()
        target = directory / key
        target.mkdir(exist_ok=True)
        src = target / f"{name}.c"
        src.write_text(source)
        library = target / f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"
        if not library.is_file():
            library = Path(
                build._build(
                    name,
                    str(src),
                    str(target),
                    driver.library_dirs(),
                    driver.include_dirs,
                    driver.libraries,
                    [],
                )
            )
        version_info = subprocess.run(
            ["readelf", "--version-info", str(library)], capture_output=True, text=True, check=True
        ).stdout
        glibc_versions.update(re.findall(r"\bGLIBC_([0-9.]+)\b", version_info))
        entries.append(
            {
                "key": key,
                "filename": library.name,
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
        src.unlink()
    metadata["modules"] = entries
    metadata["inventory"] = inventory
    metadata["glibc_min"] = (
        max(glibc_versions, key=lambda v: tuple(map(int, v.split(".")))) if glibc_versions else None
    )
    installed = distribution("triton")
    licenses = [path for path in installed.files or [] if str(path).endswith("licenses/LICENSE")]
    if not licenses:
        raise RuntimeError("Triton installation is missing its redistribution license")
    (directory / "LICENSE.triton").write_bytes(
        Path(installed.locate_file(licenses[0])).read_bytes()
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def install() -> bool:
    """Install compatible bundled host modules into Triton's normal cache.

    Returns
    -------
    bool
        True when a bundle is available for the current cache; False for source
        installations, which retain Triton's ordinary compilation behavior.

    Raises
    ------
    ImportError
        Python/Triton/source identity, GLIBC or any module checksum is incompatible.
        All checksums are validated before the first cache entry is written.

    Notes
    -----
    Installation is thread safe and tracks changes to the configured cache.
    Existing Triton cache-manager and compiler hooks are preserved.
    """
    with _lock:
        return _install()


def _install() -> bool:
    """Validate and copy a bundled host-launcher set under the caller's lock.

    Return False when no manifest exists, or True after this Triton cache identity
    has already been populated. Otherwise compare source/ABI/platform/GLIBC fields,
    read and verify every module checksum before writing any cache entry, then
    populate missing or mismatched entries through Triton's own cache manager.
    Only after all writes succeed record ``_installed_cache``. Errors propagate
    and leave the identity unset so a later attempt can retry. The public install
    wrapper owns thread synchronization; no compiler hook is replaced here.
    """
    global _installed_cache
    directory = ROOT / "_binaries" / "triton"
    manifest = directory / "metadata.json"
    if not manifest.is_file():
        return False
    from triton import knobs

    cache_identity = (knobs.cache.dir, knobs.cache.manager_class)
    if _installed_cache == cache_identity:
        return True
    metadata = json.loads(manifest.read_text())
    for field, value in identity().items():
        if metadata.get(field) != value:
            raise ImportError(
                f"Bundled Triton host modules require {field}={metadata.get(field)!r}; installed value is {value!r}."
            )
    libc, installed = platform.libc_ver()
    required = metadata.get("glibc_min")
    if required and (
        libc != "glibc"
        or tuple(map(int, installed.split("."))) < tuple(map(int, required.split(".")))
    ):
        raise ImportError(
            f"Bundled Triton host modules require glibc >= {required}; installed runtime is {libc} {installed}."
        )
    from triton.runtime.cache import get_cache_manager

    validated = []
    for entry in metadata["modules"]:
        path = directory / entry["key"] / entry["filename"]
        binary = path.read_bytes()
        if hashlib.sha256(binary).hexdigest() != entry["sha256"]:
            raise ImportError(f"Bundled Triton host module checksum mismatch: {path.name}")
        validated.append((entry, binary))
    for entry, binary in validated:
        cache = get_cache_manager(entry["key"])
        cached = cache.get_file(entry["filename"])
        if cached and hashlib.sha256(Path(cached).read_bytes()).hexdigest() == entry["sha256"]:
            continue
        cache.put(binary, entry["filename"], binary=True)
    _installed_cache = cache_identity
    return True
