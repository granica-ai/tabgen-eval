"""Apple Silicon build configuration; real compilation runs on the macOS CI lane."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabgen_eval.execution.torch.cpu import extension


@pytest.fixture
def mac(monkeypatch, tmp_path):
    monkeypatch.setattr(extension.sys, "platform", "darwin")
    monkeypatch.setattr(extension.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TABGEN_EVAL_OPENMP_ROOT", raising=False)
    torch_root = tmp_path / "torch"
    (torch_root / "include").mkdir(parents=True)
    (torch_root / "lib").mkdir()
    return SimpleNamespace(__file__=str(torch_root / "__init__.py")), torch_root


def openmp(prefix):
    (prefix / "include").mkdir(parents=True, exist_ok=True)
    (prefix / "include" / "omp.h").touch()
    (prefix / "lib").mkdir(exist_ok=True)
    (prefix / "lib" / "libomp.dylib").touch()


def test_bundled_openmp_does_not_need_homebrew(mac, monkeypatch):
    torch, root = mac
    openmp(root)

    def forbidden(*args, **kwargs):
        pytest.fail("Bundled OpenMP must not require brew")

    monkeypatch.setattr(extension.subprocess, "run", forbidden)
    assert extension._mac_openmp(torch) == (root / "include", root / "lib" / "libomp.dylib")


def test_custom_headers_still_use_torch_runtime_to_avoid_duplicates(mac, tmp_path, monkeypatch):
    torch, root = mac
    (root / "lib" / "libomp.dylib").touch()
    custom = tmp_path / "custom-openmp"
    openmp(custom)
    monkeypatch.setenv("TABGEN_EVAL_OPENMP_ROOT", str(custom))
    assert extension._mac_openmp(torch) == (custom / "include", root / "lib" / "libomp.dylib")


def test_custom_openmp_runtime_used_when_torch_has_none(mac, tmp_path, monkeypatch):
    torch, _ = mac
    custom = tmp_path / "custom-openmp"
    openmp(custom)
    monkeypatch.setenv("TABGEN_EVAL_OPENMP_ROOT", str(custom))
    assert extension._mac_openmp(torch) == (custom / "include", custom / "lib" / "libomp.dylib")


def test_missing_custom_openmp_has_macos_installation_instructions(mac, tmp_path, monkeypatch):
    torch, _ = mac
    monkeypatch.setenv("TABGEN_EVAL_OPENMP_ROOT", str(tmp_path / "missing"))
    with pytest.raises(ImportError, match="omp.h.*xcode-select.*brew install libomp"):
        extension._mac_openmp(torch)


@pytest.mark.parametrize("version, standard", [("2.6.0", 17), ("2.15.0", 20)])
def test_macos_flags_use_clang_and_dynamic_library_without_linux_options(mac, version, standard):
    torch, root = mac
    torch.__version__ = version
    openmp(root)
    flags, omp, link, suffix = extension._build_settings(torch)
    assert "-arch" in flags and "arm64" in flags and suffix == ".dylib"
    assert f"-std=c++{standard}" in flags
    assert "-Xpreprocessor" in omp and "-fopenmp" in omp
    assert str(root / "lib" / "libomp.dylib") in link
    assert "-march=native" not in flags
    assert "-ldl" not in link


@pytest.mark.parametrize("architecture", ["x86_64", "AMD64"])
def test_intel_and_rosetta_processes_are_rejected(mac, monkeypatch, architecture):
    monkeypatch.setattr(extension.platform, "machine", lambda: architecture)
    assert not extension._supported_platform()
    with pytest.raises(ImportError, match="native arm64 Python.*Rosetta"):
        extension._build(SimpleNamespace(), load=False)


@pytest.mark.parametrize("signing_fails", [False, True])
def test_macos_repaired_library_is_published_only_after_signing(
    tmp_path, monkeypatch, signing_fails
):
    # Import Torch's host tools on the real platform before simulating Darwin.
    torch = pytest.importorskip("torch")
    cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    monkeypatch.setattr(cpp_extension, "library_paths", lambda: [str(tmp_path)])

    monkeypatch.setattr(extension.sys, "platform", "darwin")
    monkeypatch.setattr(extension.platform, "machine", lambda: "arm64")
    runtime = tmp_path / "libomp.dylib"
    runtime.touch()
    monkeypatch.setattr(extension, "_mac_openmp", lambda _: (tmp_path, runtime))
    monkeypatch.setenv("TABGEN_EVAL_CPU_CACHE", str(tmp_path / "cache"))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).touch()
        if command[0] == "otool":
            return SimpleNamespace(
                returncode=0,
                stdout=f"{command[-1]}:\n\t/opt/llvm-openmp/lib/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)\n",
                stderr="",
            )
        if (
            signing_fails
            and command[0] == "codesign"
            and Path(command[-1]).name == "building.dylib"
        ):
            return SimpleNamespace(returncode=1, stdout="", stderr="code signing failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(extension.subprocess, "run", run)
    if signing_fails:
        with pytest.raises(RuntimeError, match="build-codesign failed"):
            extension._build(torch, load=False)
        assert not list((tmp_path / "cache").rglob("tabgen_eval_cpu.dylib"))
        return
    target = extension._build(torch, load=False)
    assert target.suffix == ".dylib" and target.is_file()
    command = next(c for c in calls if "-dynamiclib" in c)
    assert "-ldl" not in command and "-shared" not in command
    assert not any("GLIBCXX" in arg or "$ORIGIN" in arg for arg in command)
    assert "-ltorch_cpu" in command and "-lc10" in command
    assert command.count(f"-Wl,-rpath,{tmp_path}") == 1
    cxx_probe = next(
        c for c in calls if any(arg.endswith(("check-cxx17.cpp", "check-cxx20.cpp")) for arg in c)
    )
    omp_probe = next(c for c in calls if any(arg.endswith("check-openmp.cpp") for arg in c))
    assert "-Xpreprocessor" not in cxx_probe and str(runtime) not in cxx_probe
    assert "-Xpreprocessor" in omp_probe and str(runtime) in omp_probe
    repairs = [c for c in calls if c[0] == "install_name_tool"]
    assert len(repairs) == 2
    assert {Path(c[-1]).name for c in repairs} == {"check-openmp", "building.dylib"}
    assert all(c[2:4] == ["/opt/llvm-openmp/lib/libomp.dylib", str(runtime)] for c in repairs)
    assert len([c for c in calls if c[0] == "codesign"]) == 2
    assert all(c[-1] != str(runtime) for c in repairs)
    count = len(calls)
    assert extension._build(torch, load=False) == target
    assert len(calls) == count


def test_macos_relocation_does_not_resign_when_dependency_is_already_correct(
    mac,
    tmp_path,
    monkeypatch,
):
    _, root = mac
    runtime = root / "lib" / "libomp.dylib"
    target = tmp_path / "check-openmp"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=f"{target}:\n\t{runtime} (compatibility version 5.0.0, current version 5.0.0)\n",
            stderr="",
        )

    monkeypatch.setattr(extension.subprocess, "run", run)
    extension._fix_macos_openmp(target, runtime, tmp_path, stage="check-openmp", prerequisite=True)
    assert len(calls) == 1 and calls[0][0] == "otool"


@pytest.mark.skipif(sys.platform != "darwin", reason="Requires real macOS Mach-O tools")
def test_real_stale_openmp_install_name_is_repaired_before_execution(tmp_path):
    """Reproduce a Torch runtime with a build-machine LC_ID_DYLIB on Apple hardware."""
    torch = pytest.importorskip("torch")
    include, original = extension._mac_openmp(torch)
    runtime = tmp_path / "libomp.dylib"
    shutil.copy2(original, runtime)
    original_bytes = original.read_bytes()
    stale = "/opt/llvm-openmp/lib/libomp.dylib"
    subprocess.run(["install_name_tool", "-id", stale, str(runtime)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(runtime)], check=True)
    flags = ["-std=c++17", "-arch", "arm64", "-Xpreprocessor", "-fopenmp", f"-I{include}"]
    link = [str(runtime), f"-Wl,-rpath,{tmp_path}", "-Wl,-headerpad_max_install_names"]
    extension._check_prerequisites(
        extension._compiler(),
        flags,
        tmp_path,
        openmp_flags=["-Xpreprocessor", "-fopenmp", f"-I{include}"],
        openmp_link_flags=link,
    )
    assert stale in (tmp_path / "check-openmp-dependencies.log").read_text()
    assert "--sign -" in (tmp_path / "check-openmp-codesign.log").read_text()
    # Also load an actual repaired dynamic library, not only a standalone executable.
    source = tmp_path / "minimal.cpp"
    source.write_text(
        '#include <omp.h>\nextern "C" int test_threads() { return omp_get_max_threads(); }\n'
    )
    target = tmp_path / "minimal.dylib"
    subprocess.run(
        extension._compiler() + flags + ["-dynamiclib", str(source), "-o", str(target)] + link,
        check=True,
    )
    extension._fix_macos_openmp(target, runtime, tmp_path, stage="build")
    # A fresh process avoids loading a second OpenMP runtime beside Torch's original.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes, sys; lib = ctypes.CDLL(sys.argv[1]); assert lib.test_threads() > 0",
            str(target),
        ],
        check=True,
    )
    assert original.read_bytes() == original_bytes
