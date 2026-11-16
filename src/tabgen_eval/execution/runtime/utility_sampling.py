"""Assign sampled source positions to predictive training and test roles."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from tabgen_eval._typing import IntArray, SeedLike

from .sampling import FixedDraw, paired_draw_batches


@dataclass(frozen=True)
class UtilityDraw:
    """Equal-sized training pools and an independent shared real test draw."""

    generated: IntArray
    real_train: IntArray
    real_test: IntArray


def utility_point_draw(draw: FixedDraw) -> UtilityDraw:
    """Use the common evaluation reference as test and its real candidate as training."""
    return UtilityDraw(draw.generated, draw.real_candidate, draw.reference)


def utility_draws(
    draw: FixedDraw, *, n_replicates: int, seed: SeedLike, include_generated: bool
) -> Iterator[UtilityDraw]:
    """Yield fresh real roles and optionally fresh generated training subsets.

    Draws share the source-index streams used by other evaluation metrics. Real
    training and test positions are disjoint within each replicate. A fixed
    generated subset is reused when generated uncertainty is not requested.
    Bounded batches keep row-index storage independent of the replicate count.
    """
    for batch in paired_draw_batches(
        draw,
        n_replicates=n_replicates,
        seed=seed,
        batch_size=8,
        include_generated=include_generated,
    ):
        if include_generated:
            assert batch.generated is not None
        for index in range(len(batch.reference)):
            yield UtilityDraw(
                batch.generated[index] if batch.generated is not None else draw.generated,
                batch.real_candidate[index],
                batch.reference[index],
            )
