"""Fetch any OpenML dataset as a real ``(train, holdout)`` split for :func:`evaluate`.

A thin on-ramp so a wider audience can try ``tabgen-eval`` out of the box: name an OpenML
dataset (or give its numeric id) and get back two real frames. The package is built around an
*unseen holdout* — the real–real ideal value only means something when ``holdout`` is a sample the
generator never saw — so this loader returns a deterministic, seeded split that encodes the
intended workflow:

    train your generator on ``train``  →  evaluate ``generated`` against ``holdout``.

The caller prepares values and supplies an explicit schema to :func:`evaluate`.
This module only downloads and splits the original values.

The fetch and split use the existing scikit-learn dependency (``fetch_openml`` +
``train_test_split``). The network call is isolated in the small
``_sklearn_fetch`` helper, which tests monkeypatch to run offline.
"""

from __future__ import annotations

import pandas as pd


def _sklearn_fetch(dataset: str | int, version: str | int, cache_dir: str | None) -> pd.DataFrame:
    """Fetch one OpenML dataset as a single frame (features + target) via scikit-learn.

    ``fetch_openml`` resolves a numeric ``data_id`` directly, or a string ``name`` at a given
    ``version`` ("active" = the current default version). ``.frame`` returns the full table.
    """
    from sklearn.datasets import fetch_openml

    key = str(dataset).strip()
    if key.isdigit():
        bunch = fetch_openml(data_id=int(key), as_frame=True, parser="auto", data_home=cache_dir)
    else:
        bunch = fetch_openml(
            name=key, version=version, as_frame=True, parser="auto", data_home=cache_dir
        )
    return bunch.frame.copy()


def load_openml(
    dataset: str | int,
    *,
    test_size: float = 0.5,
    seed: int = 42,
    shuffle: bool = True,
    version: str | int = "active",
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch an OpenML dataset (by name or numeric id) as a real ``(train, holdout)`` split.

    Both frames are *real* data. Train your generator on ``train``; evaluate your generated
    frame against the unseen ``holdout`` — the proxy for "real" that the real–real ideal value is
    computed against::

        from tabgen_eval import EvalData, Schema, evaluate, load_openml
        import pandas as pd

        train, holdout = load_openml("adult")     # or load_openml(45068)
        # ... train your generator on `train`, sample `generated` from it ...
        generated = pd.read_csv("my_synthetic_adult.csv")
        schema = Schema.infer(train)  # Inspect and correct the proposed roles.
        print(evaluate(EvalData(test=holdout, generated=generated, schema=schema),
                       group=["fidelity"]).summary())

    The split is deterministic given ``(dataset, test_size, seed, shuffle)``, matching the
    package's determinism contract.

    Args:
        dataset: an OpenML dataset name (e.g. ``"adult"``) or numeric data id (``45068`` /
            ``"45068"``).
        test_size: fraction of rows held out as the ``holdout`` (the rest is ``train``).
        seed: RNG seed for the split (``random_state``).
        shuffle: shuffle rows before splitting (recommended; off preserves row order).
        version: dataset version when resolving by name (``"active"`` = the active version);
            ignored when ``dataset`` is a numeric id.
        cache_dir: where ``fetch_openml`` caches downloads (its default ``~/scikit_learn_data``
            if ``None``).

    Returns:
        ``(train, holdout)`` — two real DataFrames with reset indices. Each holds the full table
        (features + target column). Callers inspect ``Schema.infer`` or declare
        roles directly, then pass that schema explicitly through ``EvalData``.
    """
    from sklearn.model_selection import train_test_split

    df = _sklearn_fetch(dataset, version, cache_dir)
    train, holdout = train_test_split(df, test_size=test_size, random_state=seed, shuffle=shuffle)
    return train.reset_index(drop=True), holdout.reset_index(drop=True)
