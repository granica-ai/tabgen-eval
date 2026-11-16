"""Independent, reproducible row budgets for training-set memorization metrics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from tabgen_eval._typing import IntArray, SeedLike


def sample_limit(value: object) -> int | None:
    """Validate a positive row cap, or None for an uncapped pool."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_samples must be a positive integer or None")
    return int(value)


@dataclass(frozen=True)
class MemorizationDraw:
    """Independent source positions, preserving every row when below the cap."""

    training: IntArray
    generated: IntArray
    n_training: int
    n_generated: int
    max_samples: int | None

    def __post_init__(self) -> None:
        for name in ("training", "generated"):
            positions = np.array(getattr(self, name), dtype=np.int64, copy=True)
            positions.setflags(write=False)
            object.__setattr__(self, name, positions)


def memorization_draw(
    n_training: int,
    n_generated: int,
    *,
    max_samples: int | None = None,
    seed: SeedLike = 42,
) -> MemorizationDraw:
    """Sample each pool uniformly without replacement using separate RNG streams.

    A smaller generated pool never reduces the training budget, or conversely.
    Source sizes refer to eligible complete-case pools. Filtering precedes this
    optional cap, so every selected row is usable. Uncapped pools retain source order.
    """
    cap = sample_limit(max_samples)
    sizes = (n_training, n_generated)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in sizes):
        raise ValueError("pool sizes must be nonnegative integers")
    streams = np.random.SeedSequence(seed).spawn(2)
    draws = [
        np.arange(n, dtype=np.int64)
        if cap is None or n <= cap
        else np.random.default_rng(stream).choice(n, cap, replace=False)
        for n, stream in zip(sizes, streams, strict=True)
    ]
    return MemorizationDraw(draws[0], draws[1], int(n_training), int(n_generated), cap)
