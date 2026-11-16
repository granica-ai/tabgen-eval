"""One C++/CUDA source inventory for builds, cache identity and binary packaging.

Backend translation units live below ``cpu`` or ``cuda``. Backend-independent
headers live below ``common``. Paths in fingerprints are relative to the C++/CUDA source root,
so equally named files in different domains remain distinct dependencies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

CSRC_ROOT = Path(__file__).parents[1] / "csrc"
_HEADERS = {".h", ".hpp", ".cuh", ".inc"}


@dataclass(frozen=True)
class SourceInventory:
    """Immutable paths describing one backend's native build dependencies.

    ``compilation`` contains sorted host C++ translation units, ``headers`` contains
    backend and shared headers, and ``device`` contains NVRTC CUDA sources (empty
    for CPU). All paths are under ``root``. These groups distinguish compilation
    inputs from runtime device sources while keeping both in artifact fingerprints.
    This record snapshots paths, not file contents or filesystem immutability.
    """

    root: Path
    compilation: tuple[Path, ...]
    headers: tuple[Path, ...]
    device: tuple[Path, ...]

    @property
    def files(self) -> tuple[Path, ...]:
        """Return all host, header, and device paths in deterministic sorted order."""
        return tuple(sorted((*self.compilation, *self.headers, *self.device)))


def inventory(kind: str, *, root: Path | None = None) -> SourceInventory:
    """Enumerate one backend plus its common native source dependencies.

    ``kind`` must be ``"cpu"`` or ``"cuda"``; otherwise raise ``ValueError``.
    ``root`` overrides the native tree for packaging checks/tests. Discovery is
    recursive, excludes unrelated suffixes, and returns deterministic path groups;
    it imports no tensor libraries and performs no compilation.
    """
    if kind not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown native backend: {kind!r}")
    root = CSRC_ROOT if root is None else Path(root)
    domains = (root / kind, root / "common")
    files = tuple(path for domain in domains for path in domain.rglob("*") if path.is_file())
    return SourceInventory(
        root,
        tuple(sorted(path for path in files if path.suffix == ".cpp")),
        tuple(sorted(path for path in files if path.suffix in _HEADERS)),
        tuple(sorted(path for path in files if kind == "cuda" and path.suffix == ".cu")),
    )


def source_digest(kind: str, *, root: Path | None = None) -> str:
    """Hash the inventory implementation and backend dependencies by path and bytes.

    Relative paths distinguish equal filenames in different domains. Both shared
    headers and CUDA device sources participate, so changes invalidate every owning
    artifact. Raw bytes include documentation; even a comment-only edit can require
    a new build cache entry or packaged source manifest. Missing files raise their
    normal filesystem errors rather than yielding an incomplete identity.
    """
    sources = inventory(kind, root=root)
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in sources.files:
        digest.update(path.relative_to(sources.root).as_posix().encode())
        digest.update(b"\0" + path.read_bytes())
    return digest.hexdigest()


def device_source(name: str) -> Path:
    """Locate the packaged NVRTC source for a supported CUDA device module.

    ``name`` selects ``"saga"`` or ``"newton"``; unknown names raise ``ValueError``.
    The returned path is not read or compiled here and existence is checked by the
    consumer. Keeping lookup here gives runtime and packaging one source owner.
    """
    if name not in {"saga", "newton"}:
        raise ValueError(f"Unknown native device source: {name!r}")
    return CSRC_ROOT / "cuda" / "classifier" / f"{name}.cu"
