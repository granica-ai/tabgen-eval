"""Python execution dependency boundaries and optional-engine import isolation."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tabgen_eval.execution import python

pytestmark = pytest.mark.unit


def test_python_implementations_do_not_depend_on_api_dispatch_or_estimators():
    root = Path(python.__file__).parent
    forbidden = (
        "tabgen_eval.metrics",
        "tabgen_eval.execution.runtime.dispatch",
        "tabgen_eval._evaluation",
        "tabgen_eval.uncertainty",
        "tabgen_eval.report",
    )
    violations = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).with_suffix("")
        package_parts = relative.parts[:-1]
        package = ".".join((python.__name__, *package_parts))

        # Type-only references document boundaries without creating runtime
        # dependencies on metric adapters or orchestration.
        def runtime_nodes(node):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                for child in node.orelse:
                    yield from runtime_nodes(child)
                return
            yield node
            for child in ast.iter_child_nodes(node):
                yield from runtime_nodes(child)

        for node in runtime_nodes(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                imports = [importlib.util.resolve_name(name, package) if node.level else name]
            for name in imports:
                if name == "tabgen_eval" or any(
                    name == prefix or name.startswith(prefix + ".") for prefix in forbidden
                ):
                    violations.append(f"{relative}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)


def test_python_modules_import_without_loading_optional_or_tensor_engines():
    code = """
import importlib
import sys
for name in (
    'stats', 'primitives', 'fidelity', 'conditional', 'distance',
    'classifier', 'utility', 'encoding.features', 'encoding.distances',
):
    importlib.import_module('tabgen_eval.execution.python.' + name)
assert not {'torch', 'triton', 'xgboost'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, timeout=30)
