"""Select bounded source positions for fixed and repeated disjoint comparison roles."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from tabgen_eval._typing import IntArray, SeedLike, UncertaintyMode


def positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    """Validate integral sampling counts without accepting booleans or floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class SamplingPlan:
    """Full source size and automatic per-side resolution.

    Every draw selects ``2 * n_res`` distinct positions from the full source.
    An odd source leaves one row unused within a draw; all rows remain eligible.
    """

    n_source: int
    n_res: int

    @classmethod
    def create(cls, n_source: int) -> SamplingPlan:
        """Use half the full source size for each disjoint side."""
        n_source = positive_integer(n_source, "n_source", minimum=2)
        return cls(n_source, n_source // 2)


@dataclass(frozen=True)
class DrawBatch:
    """Consecutive replicate indices and their positional source draws.

    ``reference`` and ``candidate`` are owned int64 matrices with one row per
    comparison. Within each comparison their union has ``2 * n_res`` entries;
    source identities may recur across different comparisons.
    """

    start: int
    reference: IntArray
    candidate: IntArray


def draw_batches(
    plan: SamplingPlan,
    *,
    n_replicates: int,
    seed: SeedLike | np.random.SeedSequence,
    batch_size: int,
) -> Iterator[DrawBatch]:
    """Yield bounded independent draws, invariant to batching and prefix length.

    A private ``default_rng(SeedSequence(seed))`` selects ``2 * n_res`` ordered
    positions uniformly without replacement from the entire source for each
    replicate. Changing batch size or requesting additional replicates leaves
    an existing prefix unchanged and never changes NumPy's global RNG state.
    """
    n_replicates = positive_integer(n_replicates, "n_replicates")
    batch_size = positive_integer(batch_size, "batch_size")
    rng = np.random.default_rng(
        seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    )
    for start in range(0, n_replicates, batch_size):
        count = min(batch_size, n_replicates - start)
        reference = np.empty((count, plan.n_res), dtype=np.int64)
        candidate = np.empty_like(reference)
        for index in range(count):
            positions = rng.choice(plan.n_source, size=2 * plan.n_res, replace=False)
            reference[index] = positions[: plan.n_res]
            candidate[index] = positions[plan.n_res :]
        yield DrawBatch(start, reference, candidate)


@dataclass(frozen=True)
class FixedDraw:
    """Three equal-length source draws sharing one real reference in both scores.

    ``reference`` and ``real_candidate`` address the holdout pool and are disjoint
    source positions. ``generated`` addresses the generated pool. Duplicate record
    values remain distinct observations; only repeated source positions are excluded.
    Arrays are owned, read-only int64 vectors in their sampled order.
    """

    reference: IntArray
    real_candidate: IntArray
    generated: IntArray
    n_holdout: int
    n_generated: int
    n_res: int

    def __post_init__(self) -> None:
        for name in ("reference", "real_candidate", "generated"):
            values = np.array(getattr(self, name), dtype=np.int64, copy=True)
            if values.ndim != 1 or len(values) != self.n_res:
                raise ValueError("Fixed comparison draws must have n_res positions")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def paired_resolution(
    n_holdout: int, n_generated: int, *, uncertainty: UncertaintyMode = "none"
) -> int:
    """Resolve equal comparison sizes from eligible real and generated pools.

    Fixed and reference-only evaluation use the smaller of generated size and
    half the holdout. Candidate-only and both-side uncertainty retain that
    holdout-limited size
    when generated membership can vary; otherwise it uses half the generated
    pool. The same resolution applies to point scores and repeated comparisons.
    """
    half_holdout = n_holdout // 2
    if uncertainty in {"candidate", "both"}:
        return half_holdout if n_generated > half_holdout else n_generated // 2
    return min(n_generated, half_holdout)


def fixed_draw(
    n_holdout: int,
    n_generated: int,
    *,
    seed: SeedLike = 42,
    uncertainty: UncertaintyMode = "none",
) -> FixedDraw:
    """Uniformly choose two disjoint real draws and a matched generated draw.

    Resolution follows :func:`paired_resolution` for the requested uncertainty
    mode. Every source position remains eligible. If all generated rows are
    used their order is retained. Sampling uses private streams and neither
    metric selection nor numerical execution affects these positions. An empty
    draw records insufficient resolution rather than increasing the sample size.
    """
    plan = SamplingPlan.create(n_holdout)
    n_generated = positive_integer(n_generated, "n_generated")
    n_res = paired_resolution(plan.n_source, n_generated, uncertainty=uncertainty)
    real_seed, generated_seed = np.random.SeedSequence(seed).spawn(2)
    positions = np.random.default_rng(real_seed).choice(plan.n_source, 2 * n_res, replace=False)
    generated = (
        np.arange(n_generated, dtype=np.int64)
        if n_generated == n_res
        else np.random.default_rng(generated_seed).choice(n_generated, n_res, replace=False)
    )
    return FixedDraw(
        positions[:n_res],
        positions[n_res:],
        generated,
        plan.n_source,
        n_generated,
        n_res,
    )


@dataclass(frozen=True)
class PairedDrawBatch:
    """Bounded paired comparisons with a shared real reference in each row."""

    start: int
    reference: IntArray
    real_candidate: IntArray
    generated: IntArray | None


def paired_draw_batches(
    draw: FixedDraw,
    *,
    n_replicates: int,
    seed: SeedLike,
    batch_size: int,
    include_generated: bool,
) -> Iterator[PairedDrawBatch]:
    """Draw fresh disjoint real roles and optional generated subsets per replicate.

    All source positions remain eligible. Private real and generated streams
    make existing prefixes independent of batch size, requested replicate count,
    and metric selection for a given resolution. Including generated draws does
    not consume the real stream or change its sampled positions.
    """
    plan = SamplingPlan(draw.n_holdout, draw.n_res)
    real_seed = np.random.SeedSequence(seed, spawn_key=(0xB007, 0))
    generated_seed = np.random.SeedSequence(seed, spawn_key=(0xB007, 1))
    generated_rng = np.random.default_rng(generated_seed)
    for batch in draw_batches(
        plan, n_replicates=n_replicates, seed=real_seed, batch_size=batch_size
    ):
        generated = None
        if include_generated:
            generated = np.empty_like(batch.reference)
            for row in generated:
                row[:] = (
                    np.arange(draw.n_generated, dtype=np.int64)
                    if draw.n_generated == draw.n_res
                    else generated_rng.choice(draw.n_generated, draw.n_res, replace=False)
                )
        yield PairedDrawBatch(batch.start, batch.reference, batch.candidate, generated)
