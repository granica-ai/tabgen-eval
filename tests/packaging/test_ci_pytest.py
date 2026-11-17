"""CI diagnostics retain subprocess failures, signals, output, and safe annotations."""

import json
import runpy
import subprocess
import sys

import pytest


@pytest.fixture
def runner(checkout_root):
    return runpy.run_path(str(checkout_root / "tools" / "run_ci_pytest.py"))


def test_success_streams_stdout_and_stderr_without_annotations(runner, capsys):
    script = "import sys; print('stdout', flush=True); print('stderr', file=sys.stderr, flush=True)"
    assert runner["run"]([sys.executable, "-c", script]) == 0
    assert capsys.readouterr().out == "stdout\nstderr\n"


def test_failure_retains_exit_status_and_escapes_command_data(runner, capsys):
    script = "print('Traceback (most recent call last):\\nValueError: 50% complete'); raise SystemExit(7)"
    assert runner["run"]([sys.executable, "-c", script]) == 7
    output = capsys.readouterr().out
    assert "ValueError: 50% complete\n" in output
    assert "::error::pytest exited 7; trace 1/1%0ATraceback" in output
    assert "%0AValueError: 50%25 complete" in output
    assert runner["escape_annotation"]("50%\r\n") == "50%25%0D%0A"


def test_versions_are_available_in_failure_annotations(runner, capsys):
    versions = '{"python": "3.13", "torch": "2.11", "xgboost": "3.4", "numpy": "2.5"}'
    assert runner["run"]([sys.executable, "-c", "raise SystemExit(2)"], versions=versions) == 2
    output = capsys.readouterr().out
    assert f"CI runtime versions: {versions}\n" in output
    assert f"::error::{versions}%0Apytest exited 2;" in output


def test_version_metadata_does_not_import_backends(checkout_root):
    script = (
        "import json, runpy, sys; "
        f"tool = runpy.run_path({str(checkout_root / 'tools' / 'run_ci_pytest.py')!r}); "
        "print(tool['runtime_versions']()); "
        "assert 'torch' not in sys.modules and 'xgboost' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30
    )
    assert set(json.loads(result.stdout)) == {"python", "torch", "xgboost", "numpy"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fatal-signal status")
def test_fatal_signal_maps_to_shell_status_and_keeps_trace(runner, capsys):
    script = (
        "import faulthandler, os, resource, signal; "
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
        "faulthandler.enable(); os.kill(os.getpid(), signal.SIGSEGV)"
    )
    assert runner["run"]([sys.executable, "-c", script]) == 139
    output = capsys.readouterr().out
    assert "Fatal Python error: Segmentation fault" in output
    assert "::error::pytest exited 139; trace 1/1%0AFatal Python error:" in output


@pytest.mark.parametrize("prefix", ["", ".."])
def test_first_fatal_trace_survives_recursive_abort(runner, capsys, prefix):
    script = (
        f"print({prefix!r} + 'Fatal Python error: Segmentation fault', flush=True); "
        "print('first_test_frame', flush=True); "
        "print('Fatal Python error: Aborted', flush=True); "
        "print('recursive_abort_frame', flush=True); "
        "raise SystemExit(139)"
    )
    assert runner["run"]([sys.executable, "-c", script]) == 139
    output = capsys.readouterr().out
    assert output.startswith(prefix + "Fatal Python error: Segmentation fault\n")
    annotations = [line for line in output.splitlines() if line.startswith("::error::")]
    assert len(annotations) == 1
    assert "trace 1/1%0AFatal Python error: Segmentation fault%0Afirst_test_frame" in annotations[0]
    assert annotations[0].index("Segmentation fault") < annotations[0].index("Aborted")


def test_trace_annotations_stay_within_runner_limits(runner, capsys):
    script = (
        "print('Fatal Python error: simulated failure', flush=True); "
        "print('frame % ' + '\U0001f40d' * 14000, flush=True); "
        "print('Extension modules: ' + 'module,' * 14000, flush=True); "
        "raise SystemExit(3)"
    )
    assert runner["run"]([sys.executable, "-c", script], versions="\U0001f40d" * 180) == 3
    output = capsys.readouterr().out
    assert "module," * 14000 in output  # Full output remains in the live job log.
    annotations = [
        line[len("::error::") :] for line in output.splitlines() if line.startswith("::error::")
    ]
    assert 1 < len(annotations) <= 7
    decoded = [
        message.replace("%0D", "\r").replace("%0A", "\n").replace("%25", "%")
        for message in annotations
    ]
    assert all(len(message.encode("utf-16-le")) // 2 <= 4096 for message in decoded)
    assert all("Extension modules:" not in message for message in decoded)
    assert "Trace truncated" in decoded[-1]
