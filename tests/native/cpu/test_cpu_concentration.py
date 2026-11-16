"""Independent direct-geometry oracles for native concentration reductions."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pn,pc", [(0, 2), (4, 0), (4, 3)])
def test_native_counts_and_pairs(dtype, pn, pc):
    import torch

    from tabgen_eval.execution.torch.cpu.extension import library

    rng = np.random.default_rng(55)
    n = 137
    numeric = rng.normal(size=(n, pn)).astype(dtype)
    categorical = rng.integers(-1, 3, size=(n, pc))
    numeric[-1], categorical[-1] = numeric[0], categorical[0]
    squared = np.zeros((n, n), dtype=dtype)
    for c in range(pn):
        delta = numeric[:, c, None] - numeric[None, :, c]
        squared += delta * delta
    cat = np.zeros_like(squared, dtype=np.int64)
    for c in range(pc):
        cat += 2 * (categorical[:, c, None] != categorical[None, :, c])
    squared += cat
    cutoffs = np.array([0.0, 0.03, 2.0, 8.0, 64.0], dtype=dtype)
    x, codes, thresholds = [torch.from_numpy(v) for v in (numeric, categorical, cutoffs)]
    before = x.clone(), codes.clone(), thresholds.clone()
    actual = library().local_concentration_counts(x, codes, thresholds, 3, 16 * 1024)
    without_self = squared.copy()
    np.fill_diagonal(without_self, np.inf)
    expected = (without_self[None, :, :] <= cutoffs[:, None, None]).sum(2)
    np.testing.assert_array_equal(actual.numpy(), expected)
    left = np.array([0, 0, 1, 3, 99], dtype=np.int64)
    right = np.array([0, n - 1, 7, 3, 12], dtype=np.int64)
    pairs = library().paired_concentration_distances(
        x, codes, torch.from_numpy(left), torch.from_numpy(right), 3
    )
    np.testing.assert_array_equal(pairs.numpy(), squared[left, right])
    for tensor, original in zip((x, codes, thresholds), before, strict=True):
        assert torch.equal(tensor, original)


def test_singletons_empty_and_global_unknown_identity():
    import torch

    from tabgen_eval.execution.torch.cpu.extension import library

    numeric = torch.empty(2, 0, dtype=torch.float64)
    codes = torch.tensor([[-1], [0]], dtype=torch.int64)
    cutoffs = torch.tensor([1.0, 2.0, float("inf")], dtype=torch.float64)
    actual = library().local_concentration_counts(numeric, codes, cutoffs, 1)
    assert actual.tolist() == [[0, 0], [1, 1], [1, 1]]
    singleton = library().local_concentration_counts(numeric[:1], codes[:1], cutoffs, 1)
    assert singleton.tolist() == [[0], [0], [0]]
    empty = library().local_concentration_counts(numeric[:0], codes[:0], cutoffs, 1)
    assert empty.shape == (3, 0)
