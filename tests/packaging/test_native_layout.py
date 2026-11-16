"""Compiled source ownership, fingerprints, and lightweight packaging imports."""

import subprocess
import sys
from pathlib import Path

import pytest

from tabgen_eval.execution.torch import sources

pytestmark = pytest.mark.unit


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_shared_header_changes_invalidate_both_backends(tmp_path):
    shared = _write(tmp_path, "common/classifier/policy.hpp", "// original policy")
    _write(tmp_path, "cpu/classifier/fit.cpp", "// cpu")
    _write(tmp_path, "cuda/classifier/fit.cpp", "// cuda")
    before = {kind: sources.source_digest(kind, root=tmp_path) for kind in ("cpu", "cuda")}
    shared.write_text("// changed common policy")
    assert all(sources.source_digest(kind, root=tmp_path) != before[kind] for kind in before)


def test_device_dependencies_invalidate_only_the_owning_backend(tmp_path):
    _write(tmp_path, "cpu/classifier/fit.cpp", "// cpu")
    device = _write(tmp_path, "cuda/classifier/fit.cu", "// original device kernel")
    before = {kind: sources.source_digest(kind, root=tmp_path) for kind in ("cpu", "cuda")}
    device.write_text("// changed device kernel")
    assert sources.source_digest("cpu", root=tmp_path) == before["cpu"]
    assert sources.source_digest("cuda", root=tmp_path) != before["cuda"]


def test_dependency_fingerprint_distinguishes_equal_names_in_different_domains(tmp_path):
    first = _write(tmp_path, "cpu/classifier/common.hpp", "// same content")
    before = sources.source_digest("cpu", root=tmp_path)
    second = tmp_path / "cpu/runtime/common.hpp"
    second.parent.mkdir(parents=True)
    first.rename(second)
    assert sources.source_digest("cpu", root=tmp_path) != before


def test_shipped_native_inventory_covers_all_cpp_and_cuda_dependencies():
    native_root = sources.CSRC_ROOT
    actual = {path for kind in ("cpu", "cuda") for path in sources.inventory(kind).files}
    expected = {
        path
        for path in native_root.rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp", ".cuh", ".inc", ".cu"}
    }
    assert actual == expected
    cpu = sources.inventory("cpu")
    cuda = sources.inventory("cuda")
    assert cpu.compilation and cpu.headers
    assert cuda.compilation and cuda.headers and cuda.device
    assert all(path.relative_to(native_root).parts[0] in {"cpu", "common"} for path in cpu.files)
    assert all(path.relative_to(native_root).parts[0] in {"cuda", "common"} for path in cuda.files)
    assert sources.device_source("saga") in cuda.device


def test_build_and_packaging_modules_import_without_tensor_engines():
    code = """
import importlib
import importlib.abc
import sys
class BlockTensorEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'xgboost'}:
            raise AssertionError('Unexpected optional-engine import: ' + fullname)
sys.meta_path.insert(0, BlockTensorEngines())
for name in (
    'sources', 'binary', 'context', 'tensors', 'cpu.extension',
    'cuda.extension', 'cuda.triton.host',
):
    importlib.import_module('tabgen_eval.execution.torch.' + name)
assert not {'torch', 'triton', 'xgboost'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, timeout=30)
