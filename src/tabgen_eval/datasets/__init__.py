"""Fetch real OpenML tables and create deterministic generator-training/test splits.

``load_openml(name_or_id)`` returns ``(train, holdout)``. Train the generator on
``train``, then compare its output with the independent ``holdout`` through
``EvalData`` and ``evaluate``. Callers prepare values and supply an explicit
schema; fetching and splitting never infer roles or clean the source values.
The loader uses scikit-learn's existing ``fetch_openml`` dependency.
"""

from __future__ import annotations

from .openml import load_openml

__all__ = ["load_openml"]
