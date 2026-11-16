"""Conditional routing policy and exact bounded histogram-bank regressions."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

conditional = pytest.importorskip("tabgen_eval.execution.torch.fidelity.conditional")
_dense_extension_wins, pair_plan = conditional._dense_extension_wins, conditional.pair_plan

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "promoted,sparse,groups,rows,scratch,expected",
    [
        (448, 1, 1, 131072, 32 * 1024**2, True),
        (448, 1, 1, 1024, 32 * 1024**2, False),
        (448, 1, 128, 131072, 32 * 1024**2, False),
        (20, 200, 1, 131072, 32 * 1024**2, False),
        (448, 1, 1, 131072, 4096, False),
        (0, 449, 1, 131072, 32 * 1024**2, False),
        (1, 0, 1, 131072, 32 * 1024**2, False),
    ],
)
def test_dense_extension_requires_scratch_and_substantial_batch_savings(
    promoted, sparse, groups, rows, scratch, expected
):
    plan = [(8192, range(promoted)), (0, range(sparse))]
    assert _dense_extension_wins(plan, groups, rows, scratch) is expected


def test_pair_plan_reuses_owned_table_and_merges_sparse_fallback():
    marginal, joint = pair_plan(2, [523, 10000], 10, "cpu")
    primary = {grid: pairs for grid, pairs in joint}
    assert len(primary[8192]) == 2
    assert len(primary[0]) == 3
    fallback = {grid: pairs for grid, pairs in joint.sparse_fallback}
    assert len(fallback[0]) == 5
    assert 8192 not in fallback
    original = {tuple(pair) for pairs in primary.values() for pair in pairs.tolist()}
    recovered = {tuple(pair) for pairs in fallback.values() for pair in pairs.tolist()}
    assert original == recovered
    assert len(recovered) == sum(len(pairs) for pairs in fallback.values())
    owners = {pairs.untyped_storage().data_ptr() for _, pairs in marginal + joint}
    owners.update(pairs.untyped_storage().data_ptr() for _, pairs in joint.sparse_fallback)
    assert len(owners) == 1


def test_huge_domains_remain_sparse_without_retaining_a_plan_cycle():
    for plan in pair_plan(2, [15000, 15000], 10, "cpu"):
        assert {grid for grid, _ in plan} == {0}
        assert plan.sparse_fallback is None
    for plan in pair_plan(1, [2, 3], 10, "cpu"):
        assert plan.sparse_fallback is None
    with pytest.raises(OverflowError, match="int64"):
        pair_plan(0, [2**32, 2**32], 10, "cpu")


@pytest.mark.parametrize(
    "pn,cards,expected",
    [
        (1, [523], True),
        (0, [523, 523], False),
        (0, [2, 3000], True),
        (0, [5000], True),
        (2, [15000, 15000], False),
        (2, [2, 3], False),
    ],
)
def test_small_pool_domains_keep_the_original_pair_budget(pn, cards, expected):
    assert conditional._extended_pairs_possible(pn, cards, 10) is expected


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("native", [False, True])
def test_large_grid_banks_match_independent_histograms(dtype, native):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    from tabgen_eval.execution.torch.cuda.fidelity.native_tv import dense_tv_batches
    from tabgen_eval.execution.torch.fidelity.grouped_ops import group_dense_tv

    rng = np.random.default_rng(1288)
    cards = [10, 523, 2, 128]
    a = np.stack([rng.integers(-1, k, (3, 2051)) for k in cards], axis=1)
    b = np.stack([rng.integers(-1, k, (3, 1031)) for k in cards], axis=1)
    a[0, 0, 0], a[0, 1, 0] = 9, 522
    a[1, 1], b[1, 1], b[2] = -1, -1, -1
    levels = np.tile(cards, (3, 1)).astype(np.int64)
    levels[1, 1] = 0
    pairs = np.array([[0, 1], [2, 1], [0, 3]], dtype=np.int64)
    args = [torch.tensor(x, device="cuda:0") for x in (a, b, pairs, levels)]
    prototype = torch.empty((), dtype=getattr(torch, dtype), device="cuda:0")
    for cond in [-1, 0, 2]:
        result = (
            torch.cat(dense_tv_batches(*args, 8192, 2, cond, False, prototype), dim=1)
            if native
            else group_dense_tv(*args, 8192, cond, False, prototype)
        )
        expected = np.full((3, len(pairs)), np.nan)
        for group in range(3):
            for q, (i, j) in enumerate(pairs):
                if cond in (i, j) or not levels[group, i] or not levels[group, j]:
                    continue
                counts = []
                for source in (a, b):
                    keep = (source[group, i] >= 0) & (source[group, j] >= 0)
                    keys = source[group, i, keep] * cards[j] + source[group, j, keep]
                    counts.append(np.bincount(keys, minlength=8192))
                if all(count.sum() for count in counts):
                    expected[group, q] = (
                        1
                        - 0.5
                        * np.abs(counts[0] / counts[0].sum() - counts[1] / counts[1].sum()).sum()
                    )
        np.testing.assert_allclose(
            result.cpu(),
            expected,
            rtol=0,
            atol=4e-6 if dtype == "float32" else 1e-12,
            equal_nan=True,
        )
    np.testing.assert_array_equal(args[0].cpu(), a)
    np.testing.assert_array_equal(args[1].cpu(), b)
