"""Host-wrapper bundles do not depend on dataset sizes or warmed device caches.

Launcher sources follow the installed Triton API, including the separate
constants map used by Triton 3.2. Portable bundle compilation targets Triton 3.6.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabgen_eval.execution.torch.cuda.triton import host as triton_host


@pytest.fixture
def make_launcher():
    """Build reference layouts with the installed driver's constants convention."""
    pytest.importorskip("triton")
    from triton.backends.nvidia import driver

    legacy = "ids" in inspect.signature(driver.make_launcher).parameters

    def source(constants, signature, declared_constants):
        if not legacy:
            return driver.make_launcher(constants, signature, None)
        constants = {
            key[0] if isinstance(key, tuple) else key: value for key, value in constants.items()
        }
        for position, kind in signature.items():
            if kind == "constexpr":
                constants.setdefault(position, 1)
        signature = {
            position: "i32" if kind == "constexpr" else kind
            for position, kind in signature.items()
            if position not in declared_constants
        }
        return driver.make_launcher(
            constants, signature, {"ids_of_const_exprs": declared_constants}
        )

    return source


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(triton_host, "ROOT", tmp_path)
    monkeypatch.setattr(triton_host, "_installed_cache", None)
    identity = {"schema": 1, "triton": "3.6.0", "python_abi": "cpython-313"}
    monkeypatch.setattr(triton_host, "identity", lambda: identity)
    cache_knobs = SimpleNamespace(dir=str(tmp_path / "cache"), manager_class=None)
    monkeypatch.setitem(
        sys.modules, "triton", SimpleNamespace(knobs=SimpleNamespace(cache=cache_knobs))
    )
    monkeypatch.setattr(triton_host.platform, "libc_ver", lambda: ("glibc", "2.35"))
    cached = {}

    class Cache:
        def __init__(self, key):
            self.key = (cache_knobs.dir, key)

        def get_file(self, name):
            return cached.get((*self.key, name))

        def put(self, data, name, binary):
            assert binary
            path = Path(cache_knobs.dir) / self.key[1] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            cached[(*self.key, name)] = str(path)

    monkeypatch.setitem(
        sys.modules, "triton.runtime.cache", SimpleNamespace(get_cache_manager=Cache)
    )
    root = tmp_path / "_binaries" / "triton"
    path = root / ("a" * 64) / "launcher.cpython-313.so"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"portable host binary")
    manifest = root / "metadata.json"
    metadata = {
        **identity,
        "glibc_min": "2.34",
        "modules": [
            {
                "key": "a" * 64,
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest.write_text(json.dumps(metadata))
    return manifest, path, cached, cache_knobs


def test_install_seeds_only_exact_validated_host_entries(bundle):
    _, path, cached, _ = bundle
    assert triton_host.install()
    assert len(cached) == 1
    assert Path(next(iter(cached.values()))).read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [("triton", "3.5.0"), ("python_abi", "cpython-312"), ("schema", 99), ("glibc_min", "2.99")],
)
def test_incompatible_host_bundle_is_rejected_before_cache_write(bundle, field, value):
    manifest, _, cached, _ = bundle
    data = json.loads(manifest.read_text())
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ImportError, match=field.replace("_min", "")):
        triton_host.install()
    assert not cached


def test_corrupt_host_binary_is_rejected(bundle):
    _, path, cached, _ = bundle
    path.write_bytes(b"changed binary")
    with pytest.raises(ImportError, match="checksum"):
        triton_host.install()
    assert not cached


def test_changing_triton_cache_root_reseeds_host_modules(bundle):
    _, _, cached, knobs = bundle
    assert triton_host.install()
    knobs.dir += "-other"
    assert triton_host.install()
    assert len(cached) == 2


def test_source_install_does_not_change_triton_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(triton_host, "ROOT", tmp_path)
    assert not triton_host.install()


def test_launcher_layouts_are_complete_and_independent_of_constexpr_values(make_launcher):
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    sources, inventory = triton_host.launcher_sources()
    assert len(inventory) == 36
    assert "classifier._hist" not in inventory
    assert "classifier._map" not in inventory
    assert "preprocessing._finite" not in inventory
    assert not any(name.startswith("grouping.") for name in inventory)
    assert "concentration._count" in inventory
    assert "concentration._pairs" in inventory
    assert "distance._self_minimum" in inventory
    assert "fidelity_grouped._divide" not in inventory
    assert len(sources) >= len(inventory)
    signature = {0: "*fp32", 1: "i32", 2: "constexpr"}
    baseline = make_launcher({(2,): 100}, signature, [2])
    assert baseline == make_launcher({(2,): 100000}, {**signature, 0: "*fp64"}, [2])
    assert baseline == make_launcher({(2,): 15000}, {**signature, 0: "*i64"}, [2])


@pytest.mark.parametrize("specialize_one", [False, True])
@pytest.mark.parametrize("feature_count", [1, 15000])
def test_launcher_sources_match_native_cuda_launcher(monkeypatch, specialize_one, feature_count):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from triton.backends.nvidia import driver

    from tabgen_eval.execution.torch.cuda.triton.distance import _tile

    legacy = "ids" in inspect.signature(driver.make_launcher).parameters
    # Model the AST metadata consumed by Triton's actual launcher constructor.
    constants = {7: feature_count, 8: 0, 9: False}
    signature = dict.fromkeys(range(5), "*fp32") | {5: "i32", 6: "u64"}
    if specialize_one:
        constants[5] = 1
    if not legacy:
        signature = {
            position: "constexpr" if position in constants else signature[position]
            for position in range(10)
        }
    captured = []

    class SourceCaptured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(kwargs["src"] if "src" in kwargs else args[0])
        raise SourceCaptured

    # Stop before compilation; this check needs neither a compiler nor a GPU.
    monkeypatch.setattr(driver, "compile_module_from_src", capture)
    monkeypatch.setattr(driver, "library_dirs", lambda: [])
    with pytest.raises(SourceCaptured):
        driver.CudaLauncher(
            SimpleNamespace(fn=_tile, signature=signature, constants=constants), None
        )
    sources, _ = triton_host.launcher_sources()
    assert captured[0] in sources


def test_grouped_runtime_strides_have_prebuilt_integer_launchers(make_launcher):
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    sources, inventory = triton_host.launcher_sources()
    sources = set(sources)
    expected = {
        "fidelity_grouped._joint": [4, 5, 6, 11],
        "fidelity_grouped._joint_banked": [4, 5, 6, 11],
        "fidelity_grouped._dense_tv": [5, 6, 8],
        "fidelity_grouped._edges": [4],
        "fidelity_grouped._center": [4],
    }
    for name, positions in expected.items():
        layout = inventory[name]
        assert layout["integers"] == positions
        for types in itertools.product(("i32", "u32", "i64", "u64"), repeat=len(positions)):
            signature = {
                i: "constexpr" if i in layout["constexpr"] else "*fp32"
                for i in range(layout["arguments"])
            }
            signature.update(zip(positions, types, strict=True))
            assert make_launcher({}, signature, layout["constexpr"]) in sources


def test_distance_presence_runtime_dimensions_have_prebuilt_launchers(make_launcher):
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    from tabgen_eval.execution.torch.cuda.triton import distance

    sources, inventory = triton_host.launcher_sources()
    sources = set(sources)
    for name, positions in {"distance._presence": [5, 6], "distance._mask": [4, 5]}.items():
        layout = inventory[name]
        assert layout["integers"] == positions
        kernel = getattr(distance, name.split(".")[1])
        assert all(kernel.params[position].do_not_specialize for position in positions)
        for types in itertools.product(("i32", "u32", "i64", "u64", "constexpr"), repeat=2):
            signature = {
                i: "constexpr" if i in layout["constexpr"] else "*fp32"
                for i in range(layout["arguments"])
            }
            signature.update(zip(positions, types, strict=True))
            assert make_launcher({}, signature, layout["constexpr"]) in sources


def test_later_corrupt_module_does_not_partially_populate_cache(bundle):
    manifest, first, cached, _ = bundle
    data = json.loads(manifest.read_text())
    later = first.parent.parent / ("b" * 64) / "later.so"
    later.parent.mkdir()
    later.write_bytes(b"corrupt binary")
    data["modules"].append({"key": "b" * 64, "filename": later.name, "sha256": "wrong"})
    manifest.write_text(json.dumps(data))
    with pytest.raises(ImportError, match="checksum"):
        triton_host.install()
    assert not cached


def test_self_distance_runtime_offsets_have_prebuilt_integer_launchers(make_launcher):
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    from tabgen_eval.execution.torch.cuda.triton.distance import _self_minimum

    sources, inventory = triton_host.launcher_sources()
    sources = set(sources)
    layout = inventory["distance._self_minimum"]
    positions = [5, 6, 10, 11]
    assert layout["integers"] == positions
    assert [_self_minimum.params[i].name for i in positions] == [
        "NQ",
        "NR",
        "QUERY_START",
        "REFERENCE_START",
    ]
    assert all(_self_minimum.params[i].do_not_specialize for i in (10, 11))
    for kinds in itertools.product(("i32", "u32", "i64", "u64", "constexpr"), repeat=4):
        signature = {
            i: "constexpr" if i in layout["constexpr"] else "*fp32"
            for i in range(layout["arguments"])
        }
        signature.update(zip(positions, kinds, strict=True))
        assert make_launcher({}, signature, layout["constexpr"]) in sources


def test_concentration_launchers_are_packaged(make_launcher):
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    sources, inventory = triton_host.launcher_sources()
    sources = set(sources)
    expected = {"concentration._count": [4], "concentration._pairs": [5]}
    for name, positions in expected.items():
        layout = inventory[name]
        assert layout["integers"] == positions
        for kinds in itertools.product(("i32", "u32", "i64", "u64"), repeat=len(positions)):
            signature = {
                i: "constexpr" if i in layout["constexpr"] else "*fp32"
                for i in range(layout["arguments"])
            }
            signature.update(zip(positions, kinds, strict=True))
            assert make_launcher({}, signature, layout["constexpr"]) in sources


def test_shared_preprocessing_launchers_cover_every_runtime_integer_layout(make_launcher):
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    from tabgen_eval.execution.torch.cuda.triton import preprocessing

    sources, inventory = triton_host.launcher_sources()
    sources = set(sources)
    runtime = {
        "_moments": ("N",),
        "_means": ("NR", "NS"),
        "_deviations": ("NR", "NS"),
        "_scales": ("N",),
        "_apply_numeric": ("N",),
        "_hist": ("N",),
        "_map": ("N",),
        "_missing": ("N",),
        "_assemble": ("N",),
        "_indicators": ("N",),
    }
    assert {name for name in inventory if name.startswith("preprocessing.")} == {
        f"preprocessing.{name}" for name in runtime
    }
    for name, scalars in runtime.items():
        kernel = getattr(preprocessing, name)
        positions = [parameter.num for parameter in kernel.params if parameter.name in scalars]
        layout = inventory[f"preprocessing.{name}"]
        strides = {"S0", "S1", "A0", "A1", "B0", "B1"}
        assert all(
            parameter.is_constexpr for parameter in kernel.params if parameter.name in strides
        )
        assert layout["integers"] == positions
        for kinds in itertools.product(
            ("i32", "u32", "i64", "u64", "constexpr"), repeat=len(positions)
        ):
            signature = {
                index: "constexpr" if index in layout["constexpr"] else "*fp32"
                for index in range(layout["arguments"])
            }
            signature.update(zip(positions, kinds, strict=True))
            assert make_launcher({}, signature, layout["constexpr"]) in sources
