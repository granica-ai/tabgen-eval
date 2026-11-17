"""Stream pytest output and expose failure traces through GitHub annotations."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import metadata

TRACE_LIMIT = 12000
CHUNK_SIZE = 1800


def escape_annotation(message: str) -> str:
    """Follow actions/toolkit's command-data escaping order."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def runtime_versions() -> str:
    """Read installed distribution metadata without importing either backend."""
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "xgboost", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return json.dumps(versions, sort_keys=True)


def run(command: list[str], *, versions: str = "") -> int:
    """Preserve command output and shell exit status, including fatal signals."""
    if versions:
        print(f"CI runtime versions: {versions}", flush=True)
    tail = fatal_trace = ""
    recording_fatal = False
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            tail = (tail + line)[-65536:]
            fatal_start = line.find("Fatal Python error:")
            if fatal_start >= 0 and not fatal_trace:
                fatal_trace = line[fatal_start:][:TRACE_LIMIT]
                recording_fatal = True
            elif recording_fatal:
                if line.startswith("Extension modules:"):
                    recording_fatal = False
                elif len(fatal_trace) < TRACE_LIMIT:
                    fatal_trace = (fatal_trace + line)[:TRACE_LIMIT]
        returncode = process.wait()
    status = returncode if returncode >= 0 else 128 - returncode
    if status:
        detail = fatal_trace or tail
        if not fatal_trace and "Traceback (most recent call last):" in detail:
            detail = detail[detail.rfind("Traceback (most recent call last):") :]
        if len(detail) >= TRACE_LIMIT:
            detail = (
                detail[:TRACE_LIMIT] + "\n[Trace truncated; full output remains in the job log.]"
            )
        detail = detail.strip() or "No failure traceback was emitted; see the job log."
        chunks = [detail[start : start + CHUNK_SIZE] for start in range(0, len(detail), CHUNK_SIZE)]
        if tail and not tail.endswith("\n"):
            print()
        # Runner issue messages are limited to 4096 UTF-16 code units. At most
        # 1800 code points plus this header fit, even for non-BMP characters.
        # The bounded trace uses at most 7 of the runner's 10 error slots.
        for index, chunk in enumerate(chunks, start=1):
            version_header = f"{versions[:180]}\n" if versions else ""
            message = (
                f"{version_header}pytest exited {status}; trace {index}/{len(chunks)}\n{chunk}"
            )
            print(f"::error::{escape_annotation(message)}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(
        run(
            [sys.executable, "-X", "faulthandler", "-m", "pytest", *sys.argv[1:]],
            versions=runtime_versions(),
        )
    )
