"""Right-closed bin membership and fidelity retain meaning in fine numeric units."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import Conditional, Trend

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("limits", [(0.0, 1.8), (-0.3, 1.5), (0.0, 0.0), (7.0, 7.0), (1e12, 1e12)])
def test_equal_width_edges_match_ieee_division_and_adjacent_bin_membership(
    torch, device, dtype, limits
):
    from tabgen_eval.execution.torch.fidelity.primitives import bin_codes, equal_width_edges

    kind = np.dtype(dtype).type
    lo, hi = map(kind, limits)
    if lo == hi:
        pad = kind(kind(abs(lo) * kind(1e-6)) + kind(1e-9))
        lo, hi = kind(lo - pad), kind(hi + pad)
    # Independent scalar arithmetic matches the native CPU edge definition.
    # An allclose check would miss changes in tied observations' bin membership.
    step = kind(kind(hi - lo) / kind(10))
    expected = np.asarray([kind(lo + kind(kind(i) * step)) for i in range(11)], dtype=dtype)
    expected[-1] = hi
    expected = expected[np.r_[True, expected[1:] != expected[:-1]]]
    source = torch.tensor(
        [*limits, np.nan, np.inf, -np.inf], dtype=getattr(torch, dtype), device=device
    )
    edges = equal_width_edges(source)
    np.testing.assert_array_equal(edges.cpu().numpy(), expected)
    assert edges.dtype == source.dtype and edges.device == source.device

    query = np.concatenate(
        (
            np.nextafter(expected, kind(-np.inf)),
            expected,
            np.nextafter(expected, kind(np.inf)),
            np.asarray([np.nan, np.inf, -np.inf], dtype=dtype),
        )
    )
    wanted = np.searchsorted(expected, query, side="left") - 1
    wanted[query == expected[0]] = 0
    wanted[~np.isfinite(query) | (query < expected[0]) | (query > expected[-1])] = -1
    actual = bin_codes(torch.tensor(query, device=device), edges)
    np.testing.assert_array_equal(actual.cpu().numpy(), wanted)


@pytest.mark.parametrize("scale", [1.0, 1e-13])
@pytest.mark.parametrize("clip", [False, True])
def test_tensor_bins_preserve_edges_adjacent_values_missingness_and_outliers(
    torch, device, dtype, scale, clip
):
    from tabgen_eval.execution.torch.fidelity.primitives import bin_codes

    kind = getattr(torch, dtype)
    edges = torch.tensor([0.0, 2.0, 4.0], dtype=kind, device=device) * scale
    middle = edges[1]
    before = torch.nextafter(middle, middle.new_tensor(-float("inf")))
    after = torch.nextafter(middle, middle.new_tensor(float("inf")))
    query = torch.stack(
        (
            -middle,
            edges[0],
            before,
            middle,
            after,
            edges[-1],
            edges[-1] + middle,
            middle.new_tensor(float("nan")),
            middle.new_tensor(-float("inf")),
            middle.new_tensor(float("inf")),
        )
    )
    original = query.clone()
    # Intervals are [0, 2*scale] and (2*scale, 4*scale]. Finite outliers
    # enter the nearest edge bin only when clipping is requested.
    expected = [0 if clip else -1, 0, 0, 0, 1, 1, 1 if clip else -1, -1, -1, -1]
    np.testing.assert_array_equal(bin_codes(query, edges, clip=clip).cpu(), expected)
    if clip:
        with_infinite = expected[:-2] + [0, 1]
        np.testing.assert_array_equal(
            bin_codes(query, edges, clip=True, clip_infinite=True).cpu(), with_infinite
        )
    if str(device).startswith("cuda"):
        from tabgen_eval.execution.torch.fidelity.grouped_ops import group_bins

        # Conditional within-slice contingencies use the grouped Triton kernel.
        grouped = group_bins(
            query[None, None],
            edges[None, None],
            torch.tensor([[2]], dtype=torch.int64, device=device),
            clip,
        )
        np.testing.assert_array_equal(grouped.cpu().ravel(), expected)
    torch.testing.assert_close(query, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("identical", [True, False])
def test_trend_and_conditional_preserve_identity_and_scale_invariance(backend_precision, identical):
    backend, dtype = backend_precision
    options = {"device": backend, "dtype": dtype}
    schema = Schema(numeric=["x"], categorical=["group", "label"])
    # Every categorical slice contains a varying numeric axis paired with a
    # categorical axis, exercising within-slice contingency binning as well as
    # global Trend and numeric conditioning labels. Rare slices are retained.
    rows = [
        (x, group, str(x % 2))
        for x in range(5)
        for group, copies in (("common", 9), ("rare", 1))
        for _ in range(copies)
    ]
    source = pd.DataFrame(rows, columns=["x", "group", "label"])
    source.loc[len(source)] = [np.nan, "common", "0"]
    source.loc[len(source)] = [2.0, None, "0"]
    candidate = source.copy(deep=True)
    if not identical:
        candidate.loc[candidate.x == 4, "label"] = "1"
    metrics = [
        Trend(**options),
        Conditional(min_count=1, num_bins=2, rare_threshold=0.3, **options),
    ]
    baseline = None
    for scale in (1.0, 1e-13):
        reference, generated = source.copy(deep=True), candidate.copy(deep=True)
        reference["x"] *= scale
        generated["x"] *= scale
        values = [
            value
            for metric in metrics
            for value in metric.compute(reference, generated, schema, rng=None).values()
        ]
        assert np.isfinite(values).all()
        if identical:
            np.testing.assert_allclose(values, np.ones(len(values)), rtol=0, atol=3e-6)
        elif baseline is None:
            assert min(values) < 0.99
        if baseline is not None:
            np.testing.assert_allclose(values, baseline, rtol=0, atol=3e-6)
        baseline = values
