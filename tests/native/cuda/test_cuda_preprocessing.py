"""GPU preprocessing against independent finite-data and category oracles."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def test_numeric_training_state_and_fused_application(torch, device, dtype):
    from tabgen_eval.preprocessing import cuda

    real = np.array([[1, np.nan, np.nan, 7], [3, np.nan, np.nan, 7]], dtype=dtype)
    generated = np.array([[5, 2, np.nan, 7], [np.inf, 4, np.nan, 7]], dtype=dtype)
    r, s = [torch.tensor(v, device=device) for v in (real, generated)]
    state = cuda.fit_numeric(r, s, linear=True)
    keep, indicators, impute, center, scale = state
    np.testing.assert_array_equal(keep.cpu(), [0, 1, 3])
    np.testing.assert_array_equal(indicators.cpu(), [0, 1])
    expected_impute = np.array([2, 3, 7], dtype=dtype)
    np.testing.assert_array_equal(impute.cpu(), expected_impute)
    train = np.concatenate((real[:, [0, 1, 3]], generated[:, [0, 1, 3]]))
    train = np.where(np.isfinite(train), train, expected_impute).astype(np.float64)
    expected_center = train.mean(0)
    expected_scale = train.std(0)
    expected_scale[expected_scale == 0] = 1
    atol = 2e-7 if dtype == "float32" else 1e-14
    np.testing.assert_allclose(center.cpu(), expected_center, atol=atol, rtol=atol)
    np.testing.assert_allclose(scale.cpu(), expected_scale, atol=atol, rtol=atol)
    heldout = np.array([[np.nan, 8, 12, 7], [9, np.inf, 12, np.nan]], dtype=dtype)
    actual = cuda.apply_numeric(torch.tensor(heldout, device=device), *state, linear=True)
    values = heldout[:, [0, 1, 3]]
    values = np.where(np.isfinite(values), values, expected_impute)
    # Cast parameters before application, matching the selected design precision.
    expected = np.column_stack(
        (
            (values - expected_center.astype(dtype)) / expected_scale.astype(dtype),
            ~np.isfinite(heldout[:, [0, 1]]),
        )
    ).astype(dtype)
    np.testing.assert_allclose(actual.cpu(), expected, atol=atol, rtol=atol)
    assert actual.dtype == getattr(torch, dtype)
    # Held-out missingness in column 3 creates no training feature.
    assert actual.shape[1] == 5


def test_numeric_tree_path_preserves_values_and_missingness(torch, device, dtype):
    from tabgen_eval.preprocessing import cuda

    r = torch.tensor(
        [[1, float("nan")], [float("inf"), float("nan")]],
        device=device,
        dtype=getattr(torch, dtype),
    )
    s = torch.tensor([[3, float("nan")]], device=device, dtype=r.dtype)
    state = cuda.fit_numeric(r, s, linear=False)
    assert state[0].tolist() == [0]
    assert not len(state[1])
    expected = torch.tensor([[1], [float("nan")]], device=device, dtype=r.dtype)
    torch.testing.assert_close(
        cuda.apply_numeric(r, *state, linear=False), expected, equal_nan=True
    )


def test_numeric_large_offset_and_partial_block_statistics(torch, device, dtype):
    from tabgen_eval.preprocessing import cuda

    rng = np.random.default_rng(55)
    values = (rng.normal(size=(1031, 9)) + 100000).astype(dtype)
    values[::11, 3] = np.nan
    r, s = [torch.tensor(v, device=device) for v in (values[:513], values[513:])]
    state = cuda.fit_numeric(r, s)
    reference = values.astype(np.float64)
    fill = np.nanmean(reference[:513], axis=0)
    reference = np.where(np.isfinite(reference), reference, fill)
    np.testing.assert_allclose(
        state[3].cpu(),
        reference.mean(0).astype(dtype),
        rtol=0 if dtype == "float32" else 4e-15,
        atol=0,
    )
    np.testing.assert_allclose(
        state[4].cpu(),
        reference.std(0).astype(dtype),
        rtol=2e-7 if dtype == "float32" else 2e-12,
        atol=0,
    )


def test_category_union_missing_and_holdout_unknown(torch, device):
    from tabgen_eval.preprocessing import cuda

    real = torch.tensor([[0, 0], [-1, 0]], device=device)
    generated = torch.tensor([[1, 1], [1, 0]], device=device)
    state = cuda.fit_categories(real, generated, (3, 3))
    mapping, domains, features, active = state
    assert domains.tolist() == [0, 4, 8]
    assert features.tolist() == [0, 3, 5]
    assert active.tolist() == [0, 1, 2, 5, 6]
    heldout = torch.tensor([[2, 2], [-1, 1], [0, -1]], device=device)
    linear = cuda.map_categories(heldout, mapping, domains, features, linear=True)
    native = cuda.map_categories(heldout, mapping, domains, features, linear=False)
    assert linear.tolist() == [[-1, -1], [0, 4], [1, -1]]
    assert native.tolist() == [[3, 2], [0, 1], [1, 2]]
    # Mapping holdout changes neither vocabulary nor fitted feature count.
    assert features.tolist() == [0, 3, 5]


def test_empty_feature_domains(torch, device, dtype):
    from tabgen_eval.preprocessing import cuda

    values = torch.empty((3, 0), device=device, dtype=getattr(torch, dtype))
    numeric = cuda.fit_numeric(values, values)
    assert cuda.apply_numeric(values, *numeric).shape == (3, 0)
    categories = torch.empty((3, 0), device=device, dtype=torch.int64)
    mapping, domains, features, active = cuda.fit_categories(categories, categories, ())
    assert not len(mapping) and not len(active)
    assert domains.tolist() == features.tolist() == [0]
    assert cuda.map_categories(categories, mapping, domains, features).shape == (3, 0)


def test_shared_assembly_preserves_indicator_and_native_category_semantics(torch, device, dtype):
    from tabgen_eval.preprocessing import cuda

    numbers = torch.tensor([[1, 2], [3, 4]], device=device, dtype=getattr(torch, dtype))
    identities = torch.tensor([[0, 3], [-1, 2]], device=device)
    actual = cuda.assemble(numbers, identities, 4, linear=True)
    expected = torch.tensor(
        [[1, 2, 1, 0, 0, 1], [3, 4, 0, 0, 1, 0]], device=device, dtype=numbers.dtype
    )
    torch.testing.assert_close(actual, expected)
    native = cuda.assemble(numbers, identities, 4, linear=False)
    torch.testing.assert_close(native, torch.cat((numbers, identities.to(numbers.dtype)), dim=1))


def test_fitted_counts_describe_training_and_holdout_without_expanding_features(torch, device):
    from tabgen_eval.preprocessing import cuda

    real = torch.tensor([[float("nan"), 1], [2, 1]], device=device)
    generated = torch.tensor([[3, 1]], device=device)
    state, missing = cuda.fit_numeric(real, generated, return_diagnostics=True)
    assert missing.tolist() == [[1, 0], [0, 0]]
    assert state[1].tolist() == [0]
    assert cuda.count_missing(real).tolist() == [1, 0]
    r = torch.tensor([[-1], [0]], device=device)
    s = torch.tensor([[1]], device=device)
    state, missing = cuda.fit_categories(r, s, (3,), return_diagnostics=True)
    assert missing.tolist() == [[1, 0]]
    holdout = torch.tensor([[-1], [2], [1]], device=device)
    mapped, missing, unseen = cuda.map_categories(holdout, *state[:3], return_diagnostics=True)
    assert mapped.tolist() == [[0], [-1], [2]]
    assert missing.tolist() == [1]
    assert unseen.tolist() == [1]


def test_numeric_stride_specializations_preserve_source_views(torch, device, dtype):
    from tabgen_eval.preprocessing import cuda

    values = torch.arange(90, device=device, dtype=getattr(torch, dtype)).reshape(10, 9)
    values[0, 0] = float("nan")
    real, generated = values[:6:2, ::2], values[6:, ::2]
    assert not real.is_contiguous() and not generated.is_contiguous()
    state = cuda.fit_numeric(real, generated)
    contiguous = cuda.fit_numeric(real.contiguous(), generated.contiguous())
    for actual, expected in zip(state, contiguous, strict=True):
        torch.testing.assert_close(actual, expected)
    actual = cuda.apply_numeric(real, *state)
    expected = cuda.apply_numeric(real.contiguous(), *contiguous)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cuda.count_missing(real), cuda.count_missing(real.contiguous()))


@pytest.mark.parametrize("code", [-2, 2, 2**63 - 1])
def test_invalid_training_category_code_raises_without_unsafe_histogram_write(torch, device, code):
    from tabgen_eval.preprocessing import cuda

    real = torch.tensor([[0], [code]], device=device, dtype=torch.int64)
    generated = torch.tensor([[1]], device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="outside its declared domain"):
        cuda.fit_categories(real, generated, (2,))
    # A rejected input must leave the device usable.
    torch.cuda.synchronize()
    assert torch.ones(1, device=device).item() == 1


@pytest.mark.parametrize("code", [-2, 4, 2**63 - 1])
def test_invalid_indicator_index_raises_without_unsafe_design_write(torch, device, code):
    from tabgen_eval.preprocessing import cuda

    numbers = torch.zeros((2, 1), device=device)
    codes = torch.tensor([[0], [code]], device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="outside the fitted feature domain"):
        cuda.assemble(numbers, codes, 4, linear=True)
    torch.cuda.synchronize()
    assert torch.ones(1, device=device).item() == 1
