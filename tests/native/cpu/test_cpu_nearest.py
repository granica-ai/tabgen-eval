"""Independent precision and positional-exclusion oracles for CPU minima."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pn,pc", [(0, 2), (1, 0), (7, 3), (0, 0)])
@pytest.mark.parametrize("self_search", [False, True])
def test_native_minima_match_independent_oracle(dtype, pn, pc, self_search):
    import torch

    from tabgen_eval.execution.torch.distance.cpu_nearest import nearest_distances_cpu
    from tabgen_eval.execution.torch.distance.geometry import Features

    rng = np.random.default_rng(123)
    nr, nq = 273, 273 if self_search else 37
    a = rng.normal(size=(nr, pn)).astype(dtype)
    ca = rng.integers(-1, 4, size=(nr, pc))
    a[-1], ca[-1] = a[0], ca[0]
    b = a.copy() if self_search else rng.normal(size=(nq, pn)).astype(dtype)
    cb = ca.copy() if self_search else rng.integers(-1, 4, size=(nq, pc))
    reference = Features(torch.from_numpy(a), torch.from_numpy(ca))
    query = Features(torch.from_numpy(b), torch.from_numpy(cb))
    snapshot = [
        x.clone()
        for x in (reference.numeric, reference.categorical, query.numeric, query.categorical)
    ]
    result = nearest_distances_cpu(
        reference, query, exclude_self=self_search, threads=3, scratch_bytes=24 * 1024
    )
    squared = np.zeros((nq, nr), dtype=dtype)
    for c in range(pn):
        delta = b[:, c, None] - a[None, :, c]
        squared += delta * delta
    categories = np.zeros_like(squared, dtype=np.int64)
    for c in range(pc):
        u, v = cb[:, c, None], ca[None, :, c]
        categories += 2 * (u != v) - ((u < 0) != (v < 0))
    squared += categories
    if self_search:
        np.fill_diagonal(squared, np.inf)
    expected = np.sqrt(squared.min(1))
    np.testing.assert_array_equal(result.numpy(), expected)
    for x, original in zip(
        (reference.numeric, reference.categorical, query.numeric, query.categorical),
        snapshot,
        strict=True,
    ):
        assert torch.equal(x, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_singletons_empty_and_large_scale(dtype):
    import torch

    from tabgen_eval.execution.torch.distance.cpu_nearest import nearest_distances_cpu
    from tabgen_eval.execution.torch.distance.geometry import Features

    def features(values):
        return Features(
            torch.tensor(values, dtype=getattr(torch, np.dtype(dtype).name)),
            torch.empty((len(values), 0), dtype=torch.int64),
        )

    one = features([[0.0, 0.0]])
    empty = Features(one.numeric[:0], one.categorical[:0])
    assert nearest_distances_cpu(empty, one).isinf().all()
    assert nearest_distances_cpu(one, empty).numel() == 0
    assert nearest_distances_cpu(one, one, exclude_self=True).isinf().all()
    scale = 1e18 if dtype == np.float32 else 1e150
    distant = features([[scale, scale]])
    actual = nearest_distances_cpu(one, distant).item()
    expected = dtype(np.sqrt(dtype(scale) ** 2 + dtype(scale) ** 2))
    assert actual == float(expected)


def test_workspace_and_invalid_inputs():
    import torch

    from tabgen_eval.execution.torch.distance.cpu_nearest import nearest_distances_cpu
    from tabgen_eval.execution.torch.distance.geometry import Features

    f = Features(torch.zeros(4, 5, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="scratch budget"):
        nearest_distances_cpu(f, f, scratch_bytes=0)
    with pytest.raises(RuntimeError, match="positive threads"):
        nearest_distances_cpu(f, f, threads=0)
    bad = Features(f.numeric.clone(), f.categorical)
    bad.numeric[0, 1] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        nearest_distances_cpu(f, bad)
