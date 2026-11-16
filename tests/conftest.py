"""Shared fixtures: small deterministic frames + a matching schema."""

from __future__ import annotations

import pandas as pd
import pytest

from tabgen_eval import Schema
from tests.helpers import make_frame


@pytest.fixture
def schema() -> Schema:
    return Schema(
        numeric=["age", "income"],
        categorical=["job", "marital"],
    )


@pytest.fixture
def test_frame() -> pd.DataFrame:
    return make_frame(500, seed=1)


@pytest.fixture
def syn_frame() -> pd.DataFrame:
    return make_frame(400, seed=2)


@pytest.fixture(scope="session")
def checkout_root():
    """Repository tooling root; never used to locate the installed package."""
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


# Fixtures stay opt-in through test arguments; importing the plugin never probes CUDA.
pytest_plugins = ("tests.support.backends",)
