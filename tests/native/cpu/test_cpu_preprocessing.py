"""Native preprocessing contracts independent of classifier fitting."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numeric_fit_uses_training_only_and_preserves_missingness(torch, ops, dtype):
    real = torch.tensor([[1, np.nan, np.nan], [3, np.nan, np.nan]], dtype=getattr(torch, dtype))
    generated = torch.tensor([[5, 8, np.nan], [np.nan, 10, np.nan]], dtype=real.dtype)
    state = ops.preprocessing_fit_numeric(real, generated, True)
    keep, flags, impute, center, scale = state
    np.testing.assert_array_equal(keep, [0, 1])
    np.testing.assert_array_equal(flags, [0, 1])
    np.testing.assert_allclose(impute, [2, 9])
    filled = np.array([[1, 9], [3, 9], [5, 8], [2, 10]], dtype=dtype)
    np.testing.assert_allclose(center, filled.mean(0), rtol=1e-6)
    np.testing.assert_allclose(scale, filled.std(0), rtol=1e-6)
    heldout = torch.tensor([[np.nan, 20, 1e12], [100, np.nan, np.nan]], dtype=real.dtype)
    originals = [x.clone() for x in state]
    transformed = ops.preprocessing_apply_numeric(heldout, *state, True)
    expected = np.column_stack(
        (
            (np.array([[2, 20], [100, 9]], dtype=dtype) - center.numpy()) / scale.numpy(),
            [[1, 0], [0, 1]],
        )
    )
    np.testing.assert_allclose(transformed, expected, rtol=1e-6)
    assert transformed.dtype == real.dtype
    for value, old in zip(state, originals, strict=True):
        torch.testing.assert_close(value, old, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tree_numeric_is_unscaled_unimputed_and_owned(torch, ops, dtype):
    real = torch.tensor([[1, np.nan], [np.nan, np.inf]], dtype=getattr(torch, dtype))
    generated = torch.tensor([[9, np.nan]], dtype=real.dtype)
    state = ops.preprocessing_fit_numeric(real, generated, False)
    assert state[1].numel() == 0
    heldout = torch.tensor([[np.inf, 1], [-np.inf, 2]], dtype=real.dtype)
    transformed = ops.preprocessing_apply_numeric(real, *state, False)
    torch.testing.assert_close(transformed, real[:, :1], equal_nan=True)
    normalized = ops.preprocessing_apply_numeric(heldout, *state, False)
    assert torch.isnan(normalized).all()
    assert transformed.data_ptr() != real.data_ptr()


def test_union_vocabulary_distinguishes_missing_and_unseen(torch, ops):
    real = torch.tensor([[0, -1], [0, 1]], dtype=torch.int64)
    generated = torch.tensor([[2, 0]], dtype=torch.int64)
    cards = torch.tensor([4, 3], dtype=torch.int64)
    mapping, domains, features, active = ops.preprocessing_fit_categories(real, generated, cards)
    np.testing.assert_array_equal(domains, [0, 5, 9])
    np.testing.assert_array_equal(features, [0, 2, 5])
    np.testing.assert_array_equal(active, [1, 3, 5, 6, 7])
    holdout = torch.tensor([[2, -1], [3, 2], [-1, 1]], dtype=torch.int64)
    linear = ops.preprocessing_map_categories(holdout, mapping, domains, features, True)
    tree = ops.preprocessing_map_categories(holdout, mapping, domains, features, False)
    np.testing.assert_array_equal(linear, [[1, 2], [-1, -1], [-1, 4]])
    np.testing.assert_array_equal(tree, [[1, 0], [2, 3], [2, 2]])
    # Transforming held-out observations cannot add categories to fitted state.
    np.testing.assert_array_equal(features, [0, 2, 5])
    with pytest.raises(RuntimeError, match="cardinality"):
        ops.preprocessing_map_categories(torch.tensor([[4, 0]]), mapping, domains, features, True)


def test_precision_matching_fuses_strided_layout_and_column_rounding(torch, ops):
    source = torch.tensor(
        [[1.00000003, 2.00000006], [3.00000004, 4.00000008]], dtype=torch.float64
    ).T
    assert not source.is_contiguous()
    widths = torch.tensor([32, 64], dtype=torch.int64)
    result = ops.preprocessing_quantize(source, widths, torch.float64)
    expected = source.clone()
    expected[:, 0] = expected[:, 0].float().double()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.is_contiguous() and result.data_ptr() != source.data_ptr()
    half = ops.preprocessing_quantize(source, torch.tensor([16, 16]), torch.float32)
    torch.testing.assert_close(half, source.half().float(), rtol=0, atol=0)


def test_empty_feature_matrices_are_valid(torch, ops):
    numbers = torch.empty((4, 0))
    state = ops.preprocessing_fit_numeric(numbers, numbers[:2], True)
    assert ops.preprocessing_apply_numeric(numbers, *state, True).shape == (4, 0)
    codes = torch.empty((4, 0), dtype=torch.int64)
    mapping, domains, features, active = ops.preprocessing_fit_categories(
        codes, codes, torch.empty(0, dtype=torch.int64)
    )
    assert mapping.numel() == active.numel() == 0
    assert ops.preprocessing_map_categories(codes, mapping, domains, features, True).shape == (4, 0)


def test_dense_assembly_preserves_raw_indicators_and_native_codes(torch, ops):
    numeric = torch.tensor([[2.0], [3.0]])
    codes = torch.tensor([[0, 2], [-1, 1]], dtype=torch.int64)
    dense = ops.preprocessing_assemble(numeric, codes, 3, True)
    torch.testing.assert_close(dense, torch.tensor([[2.0, 1, 0, 1], [3, 0, 1, 0]]))
    native = ops.preprocessing_assemble(numeric, codes, 3, False)
    torch.testing.assert_close(native, torch.tensor([[2.0, 0, 2], [3, -1, 1]]))
    assert dense.dtype == native.dtype == numeric.dtype


def test_batched_linear_provenance_comes_from_fitted_training_state(torch, ops):
    real = torch.tensor(
        [[np.nan, np.nan], [1, np.nan], [2, np.nan], [3, np.nan], [4, np.nan], [5, np.nan]]
    )
    generated = torch.tensor(
        [[6, np.nan], [7, np.nan], [np.nan, np.nan], [9, np.nan], [10, np.nan], [11, np.nan]]
    )
    rc = torch.tensor([[0], [0], [-1], [3], [0], [0]], dtype=torch.int64)
    sc = torch.tensor([[2], [2], [2], [4], [-1], [2]], dtype=torch.int64)
    draw = torch.arange(6, dtype=torch.int64)[None]
    fold = draw % 3
    complete = torch.ones(6, dtype=torch.bool)
    result = ops.classifier(
        real,
        rc,
        generated,
        sc,
        complete,
        complete,
        draw,
        draw,
        fold,
        fold,
        torch.tensor([17], dtype=torch.int64),
        3,
        "direct",
        0.0,
        1e-4,
        100,
        512,
        2e11,
        1,
        8,
        2,
        10**7,
        1,
        1,
    )
    metadata = result[4][0]
    np.testing.assert_array_equal(metadata[0], [0])
    np.testing.assert_array_equal(metadata[1], [0])
    np.testing.assert_array_equal(metadata[2], [[0, -1, 0], [0, 0, 1], [0, 2, 2]])
    np.testing.assert_array_equal(metadata[3], [[0, 1, 1, 0], [4, 4, 2, 2]])
    np.testing.assert_array_equal(metadata[4], [[1, 1, 0, 0]])
    np.testing.assert_array_equal(metadata[5], [[1, 1]])


def test_preprocessing_worker_budget_is_local_and_restored_on_error(torch, ops):
    values = torch.arange(24000, dtype=torch.float32).reshape(6000, 4)
    before = torch.get_num_threads()
    one = ops.preprocessing_fit_numeric(values, values, True, 1)
    many = ops.preprocessing_fit_numeric(values, values, True, 4)
    for a, b in zip(one, many, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.get_num_threads() == before
    invalid = one[0].clone()
    invalid[0] = 9
    with pytest.raises(RuntimeError, match="column"):
        ops.preprocessing_apply_numeric(values, invalid, *one[1:], True, 2)
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError, match="thread budget"):
        ops.preprocessing_fit_numeric(values, values, True, 0)
    assert torch.get_num_threads() == before


def test_missing_counts_include_nonfinite_observations_without_replacing_values(torch, ops):
    values = torch.tensor([[1, np.nan, np.inf], [-np.inf, 2, 3]], dtype=torch.float32)
    before = values.clone()
    counts = ops.preprocessing_count_missing(values, 2)
    np.testing.assert_array_equal(counts, [1, 1, 1])
    torch.testing.assert_close(values, before, equal_nan=True)
