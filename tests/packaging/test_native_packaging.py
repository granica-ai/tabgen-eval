"""Binary-wheel compatibility and integrity checks independent of CUDA hardware."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest supplies tomli there.
    import tomli as tomllib

from tabgen_eval.execution.torch import binary as native_binary


@pytest.mark.parametrize("extra", [None, "cuda", "xgboost-cuda"])
@pytest.mark.parametrize(
    "platform,torch_version,supported",
    [
        ("linux", "2.5.0", False),
        ("linux", "2.6.0", True),
        ("linux", "2.10.0", True),
        ("linux", "2.11.0", True),
        ("darwin", "2.5.0", False),
        ("darwin", "2.6.0", True),
        ("darwin", "2.10.0", True),
        ("darwin", "2.11.0", False),
    ],
)
def test_torch_requirements_combine_core_platform_and_extras(
    checkout_root, extra, platform, torch_version, supported
):
    project = tomllib.loads((checkout_root / "pyproject.toml").read_text())["project"]
    dependencies = list(project["dependencies"])
    if extra is not None:
        dependencies.extend(project["optional-dependencies"][extra])
    environment = {"sys_platform": platform, "extra": extra or ""}
    requirements = [
        requirement
        for requirement in map(Requirement, dependencies)
        if requirement.name == "torch"
        and (requirement.marker is None or requirement.marker.evaluate(environment))
    ]
    assert requirements
    assert all(torch_version in requirement.specifier for requirement in requirements) is supported


@pytest.mark.parametrize("extra", ["cuda", "xgboost-cuda"])
@pytest.mark.parametrize(
    "torch_version,cuda_version,triton_version",
    [
        ("2.6.0+cu118", "11.8", "3.2.0"),
        ("2.6.1+cu118", "11.8", "3.2.1"),
        ("2.11.0+cu130", "13.0", "3.6.0"),
        ("2.6.0+cu124", "12.4", "3.6.0"),
        ("2.11.0+cu130", "13.0", "3.2.0"),
        ("2.12.0", "13.0", "3.7.0"),
        ("3.0.0", "14.0", "4.0.0"),
    ],
)
def test_cuda_extra_requirements_admit_runtime_supported_stacks(
    checkout_root, monkeypatch, extra, torch_version, cuda_version, triton_version
):
    from tabgen_eval.execution.torch import context

    project = tomllib.loads((checkout_root / "pyproject.toml").read_text())["project"]
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, project["optional-dependencies"][extra])
    }
    for name, installed in (("torch", torch_version), ("triton", triton_version)):
        requirement = requirements[name]
        assert installed in requirement.specifier
        assert requirement.marker is None or requirement.marker.evaluate({"sys_platform": "linux"})
    monkeypatch.setattr(context, "version", lambda name: triton_version)
    framework = SimpleNamespace(
        __version__=torch_version, version=SimpleNamespace(cuda=cuda_version)
    )
    assert context.cuda_supported(framework)


@pytest.mark.parametrize("extra", ["cuda", "xgboost-cuda"])
@pytest.mark.parametrize(
    "name,unsupported",
    [("torch", "2.5.0"), ("triton", "3.1.0")],
)
def test_cuda_extra_requirements_keep_minimum_versions(checkout_root, extra, name, unsupported):
    project = tomllib.loads((checkout_root / "pyproject.toml").read_text())["project"]
    requirement = next(
        requirement
        for requirement in map(Requirement, project["optional-dependencies"][extra])
        if requirement.name == name
    )
    assert unsupported not in requirement.specifier


def _torch():
    return SimpleNamespace(
        __version__="2.11.0+cu130",
        __file__="/torch/not/installed/__init__.py",
        version=SimpleNamespace(cuda="13.0"),
        compiled_with_cxx11_abi=lambda: True,
    )


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    monkeypatch.setattr(native_binary, "ROOT", tmp_path)
    monkeypatch.setattr(native_binary.platform, "libc_ver", lambda: ("glibc", "2.35"))
    root = tmp_path / "csrc" / "cpu" / "classifier"
    root.mkdir(parents=True)
    (root / "test.cpp").write_text("// source")
    (tmp_path / "csrc" / "cuda" / "classifier").mkdir(parents=True)
    (tmp_path / "csrc" / "common" / "classifier").mkdir(parents=True)
    (tmp_path / "csrc" / "cuda" / "classifier" / "test.cpp").write_text("// source cuda")

    def make(kind="cpu"):
        directory = tmp_path / "_binaries" / kind
        directory.mkdir(parents=True)
        library = directory / f"tabgen_eval_{kind}.so"
        library.write_bytes(b"test binary")
        metadata = native_binary.identity(_torch(), kind)
        metadata["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        metadata["glibc_min"] = "2.34"
        path = directory / "metadata.json"
        path.write_text(json.dumps(metadata))
        return library, path

    return make


@pytest.mark.parametrize("kind", ["cpu", "cuda"])
def test_compatible_binary_is_discovered(packaged, kind):
    library, _ = packaged(kind)
    assert native_binary.discover(_torch(), kind) == (library, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch", "2.10.0+cu130"),
        ("torch", "2.11.0+cpu"),
        ("cxx11_abi", False),
        ("cuda", "12.8"),
        ("machine", "unsupported"),
        ("system", "Darwin"),
        ("sources_sha256", "changed sources"),
        ("schema", 999),
    ],
)
def test_incompatible_artifact_is_rejected_before_load(packaged, field, value):
    _, path = packaged("cuda")
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    library, reason = native_binary.discover(_torch(), "cuda")
    assert library is None
    assert field in reason


def test_corrupt_binary_is_rejected(packaged):
    library, _ = packaged()
    library.write_bytes(b"corrupt")
    assert "checksum" in native_binary.discover(_torch(), "cpu")[1]


@pytest.mark.parametrize(
    "kind,suffix",
    [
        ("cpu", ".cpp"),
        ("cpu", ".h"),
        ("cpu", ".hpp"),
        ("cuda", ".cpp"),
        ("cuda", ".h"),
        ("cuda", ".hpp"),
        ("cuda", ".cu"),
    ],
)
def test_native_source_changes_reject_stale_packaged_library(packaged, kind, suffix):
    source = native_binary.ROOT / "csrc" / kind / "classifier" / f"saga{suffix}"
    original = b"// original native source\n"
    source.write_bytes(original)
    library, _ = packaged(kind)
    assert native_binary.discover(_torch(), kind) == (library, None)
    source.write_bytes(original + b"// changed inverse or device implementation\n")
    selected, reason = native_binary.discover(_torch(), kind)
    assert selected is None
    assert "sources_sha256" in reason
    source.write_bytes(original)
    assert native_binary.discover(_torch(), kind) == (library, None)


def test_glibc_requirement_is_checked(packaged):
    _, path = packaged()
    metadata = json.loads(path.read_text())
    metadata["glibc_min"] = "2.99"
    path.write_text(json.dumps(metadata))
    assert "glibc >= 2.99" in native_binary.discover(_torch(), "cpu")[1]


def test_shared_cpu_calibration_changes_reject_stale_cuda_library(packaged):
    header = native_binary.ROOT / "csrc" / "common" / "classifier" / "saga_rate.hpp"
    header.write_text("// first shared rate policy\n")
    library, _ = packaged("cuda")
    assert native_binary.discover(_torch(), "cuda") == (library, None)
    header.write_text("// changed shared rate policy\n")
    selected, reason = native_binary.discover(_torch(), "cuda")
    assert selected is None and "sources_sha256" in reason


def test_missing_metadata_is_rejected(packaged):
    _, path = packaged()
    path.unlink()
    assert "metadata is invalid" in native_binary.discover(_torch(), "cpu")[1]


def test_source_wheel_needs_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(native_binary, "ROOT", tmp_path)
    assert native_binary.discover(_torch(), "cpu") == (None, None)


def test_release_rpaths_are_relative():
    assert all(path.startswith("$ORIGIN/") for path in native_binary.release_rpaths())


@pytest.mark.parametrize("bundle_host", [False, True])
@pytest.mark.parametrize("darwin_constraint", [False, True])
def test_repacked_wheel_has_integral_record_and_exact_torch_dependency(
    tmp_path, packaged, bundle_host, darwin_constraint, checkout_root
):
    script = checkout_root / "tools" / "build_tensor_wheel.py"
    spec = importlib.util.spec_from_file_location("build_tensor_wheel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / "tabgen_eval-0.1.0-py3-none-any.whl"
    info = "tabgen_eval-0.1.0.dist-info"
    platform_dependency = 'Requires-Dist: torch<2.11; sys_platform == "darwin"\n'
    metadata = (
        "Metadata-Version: 2.4\nName: tabgen-eval\nVersion: 0.1.0\n"
        "Requires-Dist: torch>=2.6\n"
        + (platform_dependency if darwin_constraint else "")
        + 'Requires-Dist: torch>=2.6; extra == "cuda"\n'
        "Requires-Dist: numpy>=1.23\n\nDescription\n"
    )
    with zipfile.ZipFile(base, "w") as archive:
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(f"{info}/RECORD", "")
        archive.writestr("tabgen_eval/execution/_binaries/cpu/old.so", "stale")
        for source in [
            "csrc/cuda/classifier/saga.cu",
            "csrc/cuda/classifier/saga_inverse.hpp",
            "csrc/cpu/classifier/saga.h",
        ]:
            archive.writestr(f"tabgen_eval/execution/{source}", f"// packaged {source}")
    library, metadata = packaged()
    artifact = ("cpu", library, json.loads(metadata.read_text()))
    host = None
    if bundle_host:
        host = tmp_path / "triton-host"
        host.mkdir()
        (host / "metadata.json").write_text(json.dumps({"triton": "3.6.0"}))
        (host / "LICENSE.triton").write_text("MIT license notice")
        (host / "wrapper.so").write_bytes(b"host wrapper")
    result = module.repack(base, tmp_path / "dist", [artifact], "2.11.0+cu130", "2.11.0", host)
    assert "1torch2110cu130cpu" in result.name
    with zipfile.ZipFile(result) as archive:
        assert not any("old.so" in name for name in archive.namelist())
        for source in [
            "csrc/cuda/classifier/saga.cu",
            "csrc/cuda/classifier/saga_inverse.hpp",
            "csrc/cpu/classifier/saga.h",
        ]:
            assert (
                archive.read(f"tabgen_eval/execution/{source}") == f"// packaged {source}".encode()
            )
        metadata = archive.read(f"{info}/METADATA").decode()
        assert metadata.count("Requires-Dist: torch") == 2 + darwin_constraint
        assert "Requires-Dist: torch==2.11.0" in metadata
        assert 'Requires-Dist: torch>=2.6; extra == "cuda"' in metadata
        assert (platform_dependency in metadata) is darwin_constraint
        requirements = [
            Requirement(line.removeprefix("Requires-Dist: "))
            for line in metadata.split("\n\n", 1)[0].splitlines()
            if line.startswith("Requires-Dist: ")
        ]
        for extra in ("", "cuda"):
            active_torch = [
                requirement
                for requirement in requirements
                if requirement.name == "torch"
                and (
                    requirement.marker is None
                    or requirement.marker.evaluate({"sys_platform": "linux", "extra": extra})
                )
            ]
            assert active_torch
            assert all("2.11.0" in requirement.specifier for requirement in active_torch)
            for other_version in ("2.10.0", "2.12.0"):
                assert not all(
                    other_version in requirement.specifier for requirement in active_torch
                )
        assert "Requires-Dist: numpy>=1.23" in metadata
        wheel_tags = archive.read(f"{info}/WHEEL").decode()
        assert "linux_" in wheel_tags
        assert "Root-Is-Purelib: false" in wheel_tags
        if bundle_host:
            import sys

            python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            assert f"{python_tag}-{python_tag}-" in wheel_tags
            assert f"-{python_tag}-{python_tag}-" in result.name
            assert "Requires-Dist: triton==3.6.0" in metadata
            assert (
                archive.read("tabgen_eval/execution/_binaries/triton/LICENSE.triton")
                == b"MIT license notice"
            )
        else:
            assert "py3-none-linux_x86_64" in wheel_tags
            assert result.name.endswith("-py3-none-linux_x86_64.whl")
        rows = list(csv.reader(io.StringIO(archive.read(f"{info}/RECORD").decode())))
        assert len(rows) == len(archive.namelist())
        for name, digest, size in rows:
            if name.endswith("/RECORD"):
                assert digest == size == ""
                continue
            value = archive.read(name)
            assert int(size) == len(value)
            assert (
                digest
                == "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            )


@pytest.mark.parametrize("kind", ["cpu", "cuda"])
def test_packaged_loader_never_uses_compiler_or_sdk(packaged, monkeypatch, kind):
    import sys

    from tabgen_eval.execution.torch.cpu import extension as cpu_extension
    from tabgen_eval.execution.torch.cuda import extension as cuda_extension

    module = cpu_extension if kind == "cpu" else cuda_extension
    library, _ = packaged(kind)
    loaded = []
    torch = _torch()
    torch.ops = SimpleNamespace(
        load_library=loaded.append, tabgen_eval_cpu=object(), tabgen_eval_cuda=object()
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module, "_loaded", False)
    monkeypatch.delenv(f"TABGEN_EVAL_{kind.upper()}_LIBRARY", raising=False)
    monkeypatch.setenv("CXX", "/compiler/does/not/exist")

    def source_forbidden(*args, **kwargs):
        pytest.fail("Packaged binaries must not compile or locate an SDK")

    monkeypatch.setattr(module, "_build", source_forbidden)
    if kind == "cuda":
        monkeypatch.setattr(module, "_sdk", source_forbidden)
        monkeypatch.setattr(module, "_register_fake", lambda _: None)
    assert module.available()
    module.library()
    assert loaded == [str(library)]


@pytest.mark.parametrize("kind", ["cpu", "cuda"])
def test_explicit_library_overrides_packaged_discovery(packaged, tmp_path, monkeypatch, kind):
    import sys

    from tabgen_eval.execution.torch.cpu import extension as cpu_extension
    from tabgen_eval.execution.torch.cuda import extension as cuda_extension

    module = cpu_extension if kind == "cpu" else cuda_extension
    packaged(kind)
    override = tmp_path / "explicit.so"
    override.write_bytes(b"explicit")
    loaded = []
    torch = _torch()
    torch.ops = SimpleNamespace(
        load_library=loaded.append, tabgen_eval_cpu=object(), tabgen_eval_cuda=object()
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module, "_loaded", False)
    monkeypatch.setenv(f"TABGEN_EVAL_{kind.upper()}_LIBRARY", str(override))
    if kind == "cuda":
        monkeypatch.setattr(module, "_register_fake", lambda _: None)
    module.library()
    assert loaded == [str(override)]


def test_cuda_loader_resolves_split_linalg_from_installed_torch(tmp_path, monkeypatch):
    torch = _torch()
    torch.__file__ = str(tmp_path / "torch" / "__init__.py")
    linalg = Path(torch.__file__).parent / "lib" / "libtorch_cuda_linalg.so"
    linalg.parent.mkdir(parents=True)
    linalg.touch()
    calls = []
    monkeypatch.setattr(native_binary.ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))
    torch.ops = SimpleNamespace(load_library=lambda path: calls.append(path))
    native_binary.load(torch, "cuda", "custom.so")
    assert calls[0] == (str(linalg), native_binary.ctypes.RTLD_GLOBAL)
    assert calls[1] == "custom.so"


def test_incompatible_binary_reports_reason_if_source_fallback_is_missing(packaged, monkeypatch):
    import sys

    from tabgen_eval.execution.torch.cpu import extension as cpu_extension

    _, path = packaged()
    metadata = json.loads(path.read_text())
    metadata["torch"] = "2.9.0"
    path.write_text(json.dumps(metadata))
    monkeypatch.setitem(sys.modules, "torch", _torch())
    monkeypatch.setattr(cpu_extension, "_loaded", False)
    monkeypatch.delenv("TABGEN_EVAL_CPU_LIBRARY", raising=False)

    def missing_compiler(_):
        raise ImportError("compiler is missing")

    monkeypatch.setattr(cpu_extension, "_build", missing_compiler)
    with pytest.raises(
        ImportError, match="requires torch=.*2.9.0.*Source fallback failed: compiler is missing"
    ):
        cpu_extension.library()


def test_repacking_obsolete_native_payload_requires_a_fresh_python_base(tmp_path, checkout_root):
    script = checkout_root / "tools" / "build_tensor_wheel.py"
    spec = importlib.util.spec_from_file_location("build_tensor_wheel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / "tabgen_eval-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(base, "w") as archive:
        archive.writestr("tabgen_eval/_rust.abi3.so", b"obsolete native engine")
    output = tmp_path / "dist"
    with pytest.raises(ValueError, match="Obsolete native extension.*current Python build backend"):
        module.repack(base, output, [], "2.11.0")
    assert not output.exists()


def test_cpu_source_cache_invalidates_when_shared_hpp_changes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tabgen_eval.execution.torch.cpu import extension

    execution = tmp_path / "execution"
    torch_root = execution / "torch"
    cpu = torch_root / "cpu"
    source = execution / "csrc" / "cpu" / "classifier"
    source.mkdir(parents=True)
    cpu.mkdir(parents=True)
    loader = cpu / "extension.py"
    loader.write_text("# isolated build-loader identity\n")
    (torch_root / "binary.py").write_text("# isolated ABI-loader identity\n")
    (source / "minimal.cpp").write_text("// compiler is mocked\n")
    shared = execution / "csrc" / "common" / "classifier"
    shared.mkdir(parents=True)
    header = shared / "saga_rate.hpp"
    header.write_text("// first calibration implementation\n")
    from tabgen_eval.execution.torch import sources as native_sources

    monkeypatch.setattr(extension, "__file__", str(loader))
    monkeypatch.setattr(native_sources, "CSRC_ROOT", execution / "csrc")
    monkeypatch.setenv("TABGEN_EVAL_CPU_CACHE", str(tmp_path / "cache"))
    builds = []
    checks = []
    monkeypatch.setattr(
        extension, "_check_prerequisites", lambda *args, **kwargs: checks.append(args)
    )

    def compile_stub(command, **kwargs):
        target = Path(command[command.index("-o") + 1])
        target.write_bytes(b"mock native library")
        builds.append(target)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(extension.subprocess, "run", compile_stub)
    first = extension._build(torch, load=False)
    assert extension._build(torch, load=False) == first
    assert len(builds) == 1
    assert len(checks) == 1
    header.write_text("// changed calibration implementation\n")
    second = extension._build(torch, load=False)
    assert first != second and len(builds) == 2
    assert len(checks) == 2
