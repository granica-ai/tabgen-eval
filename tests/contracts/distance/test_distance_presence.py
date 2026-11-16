"""Draw-local CUDA category membership, bounded storage and exact parity."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.execution.torch.distance import executor
from tabgen_eval.execution.torch.distance.categories import CategoryDomain
from tabgen_eval.execution.torch.distance.geometry import DistanceWorkspace, ReferenceGeometry
from tabgen_eval.execution.torch.distance.ops import squared_distance_tile
from tabgen_eval.execution.torch.tensors import TensorTable
from tabgen_eval.metrics import DCR, PRDC

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")


def table(numeric, categorical, dtype="float64"):
    numeric = torch.tensor(numeric, dtype=getattr(torch, dtype), device="cuda:0")
    categorical = torch.tensor(categorical, dtype=torch.int64, device="cuda:0")
    return TensorTable(
        numeric,
        categorical,
        torch.ones(len(numeric), dtype=torch.bool, device="cuda:0"),
    )


def clone(source):
    return TensorTable(source.numeric.clone(), source.categorical.clone(), source.complete.clone())


def workspace(reference, domain, budget=16):
    context = SimpleNamespace(
        reference=reference,
        inputs=SimpleNamespace(device=reference.numeric.device, cardinalities=domain.cardinalities),
        n_numeric=reference.numeric.shape[1],
        n_categorical=reference.categorical.shape[1],
        gpu_workspace_mb=budget,
    )
    n = reference.n_rows
    return DistanceWorkspace(context, [n, n], [n, n], category_metadata_bytes=domain.metadata_bytes)


def fitted(reference, domain, budget=16):
    return ReferenceGeometry(
        reference, domain=domain, workspace=workspace(reference, domain, budget)
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "cards",
    [
        (2, 17),
        (0, 1),
        (4096, 4),
        (4097, 4),
        (1, 2),
        (4, 8),
        (16, 32),
        (64, 128),
        (256, 512),
        (1024, 2048),
    ],
)
def test_draw_local_unknown_duplicates_and_owned_buffers(dtype, cards):
    r = table(
        [[0.0, 5.0], [2.0, 5.0], [2.0, 5.0]],
        [[0 if cards[0] else -1, 0], [0 if cards[0] else -1, -1], [0 if cards[0] else -1, -1]],
        dtype,
    )
    q = table(
        [[4.0, 5.0], [-2.0, 5.0], [0.0, 5.0]],
        [[1 if cards[0] > 1 else -1, 0], [-1, -1], [0 if cards[0] else -1, 0]],
        dtype,
    )
    original_r, original_q = clone(r), clone(q)
    control = ReferenceGeometry(r)
    domain = CategoryDomain(cards, r.numeric.device)
    candidate = fitted(r, domain)
    assert (candidate.domain is None) == (max(cards) > 4096)
    a, b = control.transform(clone(r), reference=True), control.transform(clone(q))
    c, d = candidate.transform(clone(r), reference=True), candidate.transform(clone(q))
    for x, y in [(a, c), (b, d)]:
        torch.testing.assert_close(x.numeric, y.numeric, atol=0, rtol=0)
        torch.testing.assert_close(x.categorical, y.categorical, atol=0, rtol=0)
    assert d.categorical[0, 0].item() == -1
    actual = squared_distance_tile(c.numeric, d.numeric, c.categorical, d.categorical)
    expected = squared_distance_tile(a.numeric, b.numeric, a.categorical, b.categorical)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    second = candidate.transform(clone(q))
    torch.testing.assert_close(second.categorical, d.categorical, atol=0, rtol=0)
    for x, y in [(r, original_r), (q, original_q)]:
        torch.testing.assert_close(x.numeric, y.numeric, atol=0, rtol=0)
        torch.testing.assert_close(x.categorical, y.categorical, atol=0, rtol=0)


def test_workspace_fallback_skips_metadata_allocation():
    r = table([[1.0], [2.0]], [[0] * 200, [1] * 200])
    domain = CategoryDomain((4096,) * 200, r.numeric.device)
    assert workspace(r, domain, budget=1).vocabulary_bytes == 200 * 512
    candidate = fitted(r, domain, budget=1)
    assert candidate.domain is None
    assert domain._metadata is None
    assert all(
        v.untyped_storage().nbytes() == v.numel() * v.element_size() for v in candidate.vocabularies
    )


@pytest.mark.parametrize("ncols", [115, 116])
def test_wide_partial_tiles_and_histogram_merging(ncols):
    generator = torch.Generator(device="cuda:0").manual_seed(92)
    cats = torch.randint(0, 33, (2053, ncols), device="cuda:0", generator=generator)
    cats[cats == 32] = -1
    r = table(np.arange(2053)[:, None], cats.cpu().numpy())
    q = table(
        np.arange(2057)[:, None],
        torch.randint(-1, 33, (2057, ncols), device="cuda:0", generator=generator).cpu().numpy(),
    )
    control = ReferenceGeometry(r)
    candidate = fitted(r, CategoryDomain((33,) * ncols, r.numeric.device))
    a, b = control.transform(clone(q)), candidate.transform(clone(q))
    torch.testing.assert_close(a.numeric, b.numeric, atol=0, rtol=0)
    torch.testing.assert_close(a.categorical, b.categorical, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_protocol_filtering_draw_vocab_and_domain_reuse(monkeypatch, dtype):
    r = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, np.nan, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
            "c": ["a", "a", "a", "a", "shared", "shared", "a", "a", None, "shared"],
        }
    )
    s = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
            "c": ["shared", "a", "shared", "a", None, "a", "shared", "a", "shared", "a"],
        }
    )
    schema = Schema(numeric=["x"], categorical=["c"])
    ctx = context_for(DCR(device="cuda:0", dtype=dtype), r, s, schema)
    rd, sd = np.array([[0, 1, 2, 2, 3]]), np.arange(10)[None]
    dist = executor.DistanceExecutor(ctx)
    with monkeypatch.context() as mp:
        mp.setattr(executor, "ReferenceGeometry", lambda source, **kw: ReferenceGeometry(source))
        expected_dcr = dist.compute_memorization_group([DCR()], rd, sd)[0]["dcr"]
        expected_prdc = dist.compute(PRDC(nearest_k=2), rd, sd)
    before_r, before_s = clone(ctx.reference), clone(ctx.candidate)
    actual_dcr = dist.compute_memorization_group([DCR()], rd, sd)[0]["dcr"]
    metadata = dist.category_domain._metadata
    actual_prdc = dist.compute(PRDC(nearest_k=2), rd, sd)
    assert dist.category_domain._metadata is metadata
    assert actual_dcr.values.keys() == expected_dcr.values.keys()
    np.testing.assert_array_equal(
        list(actual_dcr.values.values()), list(expected_dcr.values.values())
    )
    assert actual_dcr.diagnostics == expected_dcr.diagnostics
    assert actual_dcr.stat_flags == expected_dcr.stat_flags
    for key in expected_prdc:
        np.testing.assert_array_equal(actual_prdc[key], expected_prdc[key])
    assert actual_dcr.diagnostics["usable_rows"] == {"training": 4, "generated": 8, "holdout": None}
    # A different reference draw must refit presence, despite sharing metadata.
    other_rd = np.array([[4, 5, 9]])
    samples, ws, _ = dist._samples((ctx.reference, ctx.candidate), (other_rd[0], sd[0]))
    geometry = ReferenceGeometry(samples[0], domain=dist.category_domain, workspace=ws)
    query = geometry.transform(samples[1])
    assert query.categorical[0, 0].item() >= 0  # "shared" is now observed.
    assert query.categorical[1, 0].item() == -1  # "a" is absent from this draw.
    for before, after in [(before_r, ctx.reference), (before_s, ctx.candidate)]:
        torch.testing.assert_close(before.numeric, after.numeric, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(before.categorical, after.categorical, atol=0, rtol=0)


def test_sorted_fallback_storage_and_scratch_reservation():
    n = 65539
    r = table(np.arange(n)[:, None], np.zeros((n, 4), dtype=np.int64))
    domain = CategoryDomain((5000,) * 4, r.numeric.device)
    ws = workspace(r, domain)
    # Warm ATen initialization before measuring its bounded allocations.
    fitted(r, domain)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fitted_geometry = ReferenceGeometry(r, domain=domain, workspace=ws)
    torch.cuda.synchronize()
    retained = sum(v.untyped_storage().nbytes() for v in fitted_geometry.vocabularies)
    assert retained == 4 * 8
    # This bound excludes the already persistent source table and covers the
    # fitted vocabularies plus one column's sorting/remapping scratch.
    assert torch.cuda.max_memory_allocated() - base <= n * 64 + ws.vocabulary_bytes + 16384


def test_nondefault_stream_and_metadata_are_read_only():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        r = table([[0.0], [2.0]], [[0, 0], [0, 1]])
        q = table([[4.0], [1.0]], [[1, 0], [-1, 1]])
        domain = CategoryDomain((2, 2), r.numeric.device)
        geometry = fitted(r, domain)
        before = [x.clone() for x in domain._metadata[:2]]
        before_groups = [(bins, columns.clone()) for bins, columns in domain._metadata[2]]
        actual = geometry.transform(q)
    stream.synchronize()
    torch.testing.assert_close(
        actual.categorical, torch.tensor([[-1, 0], [-1, 1]], device="cuda:0")
    )
    for a, b in zip(before, domain._metadata[:2], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for (bins, a), (other_bins, b) in zip(before_groups, domain._metadata[2], strict=True):
        assert bins == other_bins
        torch.testing.assert_close(a, b, atol=0, rtol=0)
