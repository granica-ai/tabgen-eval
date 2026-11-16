"""CPU source-build prerequisites and actionable failure diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabgen_eval.execution.torch.cpu import extension

pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="POSIX source build"
)


@pytest.mark.parametrize(
    "version, expected",
    [("2.6.0+cu118", 17), ("2.11.0", 17), ("2.12.0", 20), ("2.15.0+cpu", 20)],
)
def test_standard_matches_installed_torch_generation(version, expected):
    assert extension._cxx_standard(SimpleNamespace(__version__=version)) == expected


def test_installed_header_can_require_cxx20_even_with_older_version_metadata(tmp_path):
    root = tmp_path / "torch"
    header = root / "include" / "ATen" / "ATen.h"
    header.parent.mkdir(parents=True)
    header.write_text("#if __cplusplus < 202002L\n#error C++20 required\n#endif\n")
    torch = SimpleNamespace(__version__="2.11.0", __file__=str(root / "__init__.py"))
    assert extension._cxx_standard(torch) == 20


@pytest.mark.skipif(
    sys.platform != "linux", reason="Local GCC regression; macOS CI uses real Torch"
)
def test_cxx20_requires_headers_compile_in_the_selected_standard(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ toolchain not installed")
    root = tmp_path / "torch"
    include = root / "include"
    (include / "ATen").mkdir(parents=True)
    (include / "torch").mkdir()
    (include / "ATen" / "ATen.h").write_text(
        "template<class T> requires (sizeof(T) > 0) struct NativeHeader {};\n",
    )
    (include / "torch" / "library.h").touch()
    torch = SimpleNamespace(__version__="2.15.0", __file__=str(root / "__init__.py"))
    flags, omp, link, _ = extension._build_settings(torch)
    assert "-std=c++20" in flags
    extension._check_prerequisites(
        [compiler],
        flags,
        tmp_path,
        openmp_flags=omp,
        openmp_link_flags=link,
        torch_include_paths=[include],
    )
    assert (tmp_path / "check-cxx20-runtime.log").is_file()
    assert (tmp_path / "check-torch-headers.log").is_file()


@pytest.mark.parametrize("value", ["", "   ", '"unclosed'])
def test_invalid_compiler_setting_has_an_actionable_error(monkeypatch, value):
    monkeypatch.setenv("CXX", value)
    with pytest.raises(ImportError, match="CXX.*Install a C\\+\\+17 compiler"):
        extension._compiler()


def test_missing_compiler_is_reported_before_full_build(tmp_path):
    with pytest.raises(ImportError, match="Could not run CPU check-cxx17.*CXX"):
        extension._check_prerequisites(["/missing/cxx"], ["-std=c++17", "-fopenmp"], tmp_path)
    assert "missing/cxx" in (tmp_path / "check-cxx17.log").read_text()
    assert not (tmp_path / "check-openmp.cpp").exists()


@pytest.mark.parametrize("stage", ["check-cxx17", "check-openmp", "check-openmp-runtime"])
def test_failed_prerequisite_identifies_stage_and_keeps_diagnostics(tmp_path, monkeypatch, stage):
    def run(command, **kwargs):
        if "-o" in command:
            Path(command[-1]).touch()
        actual = (
            command[-1].split("/")[-1]
            if "-o" in command
            else command[0].split("/")[-1] + "-runtime"
        )
        return SimpleNamespace(
            returncode=int(actual == stage), stdout="", stderr="toolchain detail"
        )

    monkeypatch.setattr(extension.subprocess, "run", run)
    with pytest.raises(ImportError, match=f"CPU {stage} failed.*CXX"):
        extension._check_prerequisites(["c++"], ["-std=c++17", "-fopenmp"], tmp_path)
    assert "toolchain detail" in (tmp_path / f"{stage}.log").read_text()


def test_timeout_is_actionable_and_preserves_partial_compiler_output(tmp_path, monkeypatch):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial output", stderr=b"detail"
        )

    monkeypatch.setattr(extension.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out after 600 seconds.*host load"):
        extension._run_build_step(["c++"], tmp_path, "build", timeout=600)
    assert "partial outputdetail" in (tmp_path / "build.log").read_text()


def test_build_failure_includes_log_and_retry_advice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        extension.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="fatal error: torch header unavailable",
        ),
    )
    with pytest.raises(RuntimeError, match="failed builds are not cached.*\n.*torch header"):
        extension._run_build_step(["c++"], tmp_path, "build", timeout=600)
    assert "torch header unavailable" in (tmp_path / "build.log").read_text()


def test_unwritable_cache_has_override_instructions(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    cache = tmp_path / "not-a-directory"
    cache.touch()
    monkeypatch.setenv("TABGEN_EVAL_CPU_CACHE", str(cache))
    with pytest.raises(
        ImportError, match="Cannot write the CPU build cache.*TABGEN_EVAL_CPU_CACHE"
    ):
        extension._build(torch, load=False)


def test_toolchain_really_compiles_links_and_runs_openmp(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ toolchain not installed")
    torch = pytest.importorskip("torch")
    flags, openmp, link, _ = extension._build_settings(torch)
    extension._check_prerequisites(
        [compiler],
        flags,
        tmp_path,
        openmp_flags=openmp,
        openmp_link_flags=link,
    )
    assert (tmp_path / "check-openmp-runtime.log").is_file()
