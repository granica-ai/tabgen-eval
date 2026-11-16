#!/usr/bin/env python3
"""Build a wheel containing portable operators for the invoking Python's Torch.

Run from a Torch-equipped environment. CUDA builds also need the matching CUDA
SDK headers (CUDA_HOME or TABGEN_EVAL_CUDA_INCLUDE). Nothing is published or installed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


def repack(
    wheel,
    output,
    artifacts,
    torch_version,
    torch_distribution_version=None,
    triton_host_directory=None,
):
    """Replace native artifacts, constrain Torch, and regenerate wheel RECORD."""
    with zipfile.ZipFile(wheel) as archive:
        if any(name.startswith("tabgen_eval/_rust") for name in archive.namelist()):
            raise ValueError(
                "Obsolete native extension in base wheel; rebuild it with the current Python build backend."
            )
        contents = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/") and "/_binaries/" not in name
        }
    record = next(name for name in contents if name.endswith(".dist-info/RECORD"))
    metadata_path = next(name for name in contents if name.endswith(".dist-info/METADATA"))
    metadata = contents[metadata_path].decode()
    # PyPI CUDA wheels can have a runtime local suffix absent from dist metadata.
    dependency_version = torch_distribution_version or torch_version
    headers, separator, body = metadata.partition("\n\n")
    headers = "\n".join(
        line
        for line in headers.splitlines()
        if not (
            re.match(r"Requires-Dist: torch(?:\b|[<>=!~;(])", line, flags=re.I) and ";" not in line
        )
    )
    contents[metadata_path] = (
        headers + f"\nRequires-Dist: torch=={dependency_version}" + separator + body
    ).encode()
    if triton_host_directory is not None:
        host_metadata = json.loads((triton_host_directory / "metadata.json").read_text())
        for path in triton_host_directory.rglob("*"):
            if path.is_file():
                contents[
                    f"tabgen_eval/execution/_binaries/triton/{path.relative_to(triton_host_directory)}"
                ] = path.read_bytes()
        head, sep, body = contents[metadata_path].decode().partition("\n\n")
        contents[metadata_path] = (
            head + f"\nRequires-Dist: triton=={host_metadata['triton']}" + sep + body
        ).encode()
    manifests = []
    for kind, library, manifest in artifacts:
        prefix = f"tabgen_eval/execution/_binaries/{kind}"
        contents[f"{prefix}/tabgen_eval_{kind}.so"] = library.read_bytes()
        contents[f"{prefix}/metadata.json"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        manifests.append(manifest)
    # Host-built operators declare their actual platform, without claiming manylinux.
    # Use linux_x86_64 for host builds instead of claiming manylinux compliance.
    wheel_metadata = next(name for name in contents if name.endswith(".dist-info/WHEEL"))
    wheel_text = contents[wheel_metadata].decode()
    old_platform = wheel.stem.rsplit("-", 1)[1]
    machine = manifests[0]["machine"]
    platform_tag = f"linux_{machine}"
    contents[wheel_metadata] = wheel_text.replace(old_platform, platform_tag).encode()
    contents[wheel_metadata] = contents[wheel_metadata].replace(
        b"Root-Is-Purelib: true", b"Root-Is-Purelib: false"
    )
    output.mkdir(parents=True, exist_ok=True)
    # Build tag distinguishes wheels without claiming a shared Torch ABI.
    variant = re.sub(r"[^a-zA-Z0-9]", "", torch_version)
    stem = wheel.stem.split("-")
    if len(stem) == 6:
        stem.pop(2)
    stem.insert(
        2, f"1torch{variant}" + ("cuda" if any(a[0] == "cuda" for a in artifacts) else "cpu")
    )
    stem[-1] = platform_tag
    if triton_host_directory is not None:
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        old_tag = "-".join(stem[-3:])
        stem[-3:-1] = [python_tag, python_tag]
        contents[wheel_metadata] = (
            contents[wheel_metadata].decode().replace(old_tag, "-".join(stem[-3:])).encode()
        )
    target = output / ("-".join(stem) + ".whl")
    rows = []
    for name, value in sorted(contents.items()):
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
        rows.append((name, f"sha256={digest}", str(len(value))))
    rows.append((record, "", ""))
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    contents[record] = buffer.getvalue().encode()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(contents.items()):
            archive.writestr(name, value)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cuda", action="store_true", help="Include CUDA cuBLAS/cuSOLVER host operators"
    )
    parser.add_argument(
        "--wheel", type=Path, help="Existing Python source wheel; otherwise build with hatchling"
    )
    args = parser.parse_args()
    import torch

    from tabgen_eval.execution.torch.binary import manifest
    from tabgen_eval.execution.torch.cpu import extension as cpu_extension
    from tabgen_eval.execution.torch.cuda import extension as cuda_extension

    if sys.platform != "linux":
        parser.error("Portable operator wheel builds currently support Linux only")
    if args.cuda and (
        sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED")
    ):
        parser.error("CUDA host-wrapper wheels require the standard GIL-enabled CPython ABI")
    with tempfile.TemporaryDirectory(prefix="tabgen-eval-wheel-") as temporary:
        wheel = args.wheel
        if wheel is None:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", temporary],
                cwd=REPOSITORY,
                check=True,
            )
            wheels = list(Path(temporary).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"Expected one Python wheel; found {wheels}")
            wheel = wheels[0]
        artifacts = []
        for kind, module in [("cpu", cpu_extension)] + (
            [("cuda", cuda_extension)] if args.cuda else []
        ):
            library = module._build(torch, portable=True, load=False)
            data = manifest(torch, kind, library)
            artifacts.append((kind, library, data))
            print(f"Built {kind}: {library}", flush=True)
        host_directory = None
        if args.cuda:
            from tabgen_eval.execution.torch.cuda.triton.host import build_bundles

            host_directory = Path(temporary) / "triton"
            data = build_bundles(host_directory)
            print(
                f"Built {len(data['modules'])} Triton host wrappers for {sysconfig.get_config_var('SOABI')}",
                flush=True,
            )
        target = repack(
            wheel, args.out, artifacts, str(torch.__version__), version("torch"), host_directory
        )
        print(target)


if __name__ == "__main__":
    main()
