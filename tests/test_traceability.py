"""Failure-mode → test traceability (machine-checked).

Every failure mode the package is designed against (catalogued in ``docs/failure-modes.md``)
must map to at least one covering test, and every reference here must resolve to a real test
function — so a coverage gap, or a renamed/deleted test, fails CI instead of going unnoticed.

Each value is either a list of ``"file.py::test_name"`` references, or one of the sentinels
``DEFERRED`` (the mode lands with a planned metric) / ``DOCUMENTED`` (a usage caveat with no
code path).
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

pytestmark = pytest.mark.unit

DEFERRED = "DEFERRED"
DOCUMENTED = "DOCUMENTED"

# failure mode -> covering test reference(s)
COVERAGE: dict[str, list[str] | str] = {
    # Statistical
    "sample-size-dependence": [
        "integration/test_evaluate.py::test_evaluate_handles_different_sizes",
        "integration/test_engine.py::test_pool_sizes_define_resolution",
        "contracts/test_fixed_evaluation.py::test_fixed_draw_matches_independent_ordered_full_source_oracle",
    ],
    "duplicate-ideal-inflation": [
        "calibration/test_calibration.py::test_real_reference_draws_exclude_shared_source_positions"
    ],
    "ratio-instability": [
        "execution/python/test_stats.py::test_safe_ratio_zero_denominator_is_nan",
    ],
    "zero-dcr-baseline": ["test_dcr.py::test_duplicate_training_zero_spacing_flags_only_ratio"],
    "mismatched-calibration-geometry": [
        "test_dcr.py::test_generated_queries_cannot_change_the_training_embedding",
        "test_dcr.py::test_memorization_point_ignores_general_uncertainty_settings",
        "contracts/test_memorization_evaluation.py::test_point_scores_match_independently_sampled_training_geometry",
        "contracts/test_fixed_evaluation.py::test_fixed_draw_matches_independent_ordered_full_source_oracle",
    ],
    "boundary-saturation": ["test_properties.py::test_saturation_produces_degenerate_ci_flag"],
    "degenerate-columns": [
        "execution/python/test_primitives.py::test_correlation_similarity_constant_is_nan",
        "test_fuzz.py::test_constant_columns",
    ],
    "high-cardinality-categoricals": [
        "test_fuzz.py::test_high_cardinality_categorical",
        "test_metrics_unit.py::test_prdc_dispatches_to_structured_for_wide_onehot",
        "contracts/distance/test_torch_distances.py::test_prdc_matches_independent_onehot_oracle",
        "native/cpu/test_cpu_fidelity.py::test_shape_repeated_unequal_draws_missing_ties_and_sparse_global_vocabulary",
    ],
    "distance-boundary-rounding": [
        "contracts/distance/test_torch_distances.py::test_strict_category_boundaries_and_unknown_vs_unknown",
        "contracts/distance/test_torch_distances.py::test_cpu_cuda_reports_preserve_sampling_and_pairing",
        "execution/python/test_distances.py::test_float64_direct_differences_preserve_nearby_large_coordinates",
    ],
    "discrete-ks-ties": ["test_metamorphic.py::test_discrete_data_ideal_absorbs_ks_ties"],
    "ci-undercoverage": ["calibration/test_calibration.py::test_calib_shape_overlap_and_power"],
    "subpopulation-association-collapse": [
        "test_metamorphic.py::test_conditional_trend_catches_within_slice_break_global_trend_misses",
    ],
    "preprocessing-leakage": [
        "contracts/classifier/test_c2st.py::test_fold_preprocessing_fits_both_training_classes_but_never_scoring_rows",
        "contracts/classifier/test_c2st.py::test_resampled_classifier_roles_have_unique_disjoint_source_rows",
        "contracts/classifier/test_c2st.py::test_native_boundary_rejects_a_source_row_in_two_folds",
        "native/cuda/test_cuda_classifier.py::test_scoring_cannot_fit_transforms_and_gpu_workspace_released",
    ],
    "classifier-optimization-failure": [
        "contracts/classifier/test_c2st.py::test_finite_nonconverged_fits_remain_in_distributions",
        "contracts/classifier/test_nonlinear_c2st.py::test_native_engine_failure_is_flagged_and_failed_bootstraps_are_excluded",
        "native/cuda/test_cuda_classifier.py::test_source_fold_conflicts_memory_failures_and_nonconvergence_are_retained",
        "native/cuda/test_cuda_classifier.py::test_default_ridge_fp32_wide_direct_fit_converges",
        "native/cuda/test_cuda_classifier.py::test_native_newton_loss_difference_retains_small_armijo_changes",
        "native/cuda/test_cuda_classifier.py::test_gpu_optimizer_matches_independent_optimum",
        "native/cuda/test_cuda_newton.py::test_small_decrement_cannot_replace_full_gradient_stopping",
        "native/cuda/test_cuda_newton.py::test_native_newton_reports_final_full_gradient_after_iteration_cap",
        "native/cuda/test_cuda_newton.py::test_native_newton_nonfinite_gradient_is_an_honest_failure",
    ],
    "multiple-comparisons": DOCUMENTED,  # usage caveat, no code path
    # Engineering / in the wild
    "wrong-input-type": [
        "test_fuzz.py::test_wrong_type_raises_typed",
        "test_schema.py::test_validate_rejects_non_dataframe",
    ],
    "column-mismatch": [
        "test_fuzz.py::test_column_mismatch_raises_typed",
        "test_schema.py::test_validate_rejects_column_mismatch",
    ],
    "dtype-coercion": [
        "test_input_contract.py::test_invalid_numeric_representations_raise_before_computation"
    ],
    "nan-inf-handling": [
        "test_input_contract.py::test_invalid_numeric_representations_raise_before_computation",
        "test_fuzz.py::test_all_nan_frame",
        "test_input_contract.py::test_shape_explicitly_compares_observed_categories",
    ],
    "degenerate-sizes": [
        "test_fuzz.py::test_single_row",
        "contracts/test_paired_resampling.py::test_explicit_custom_minimum_skips_only_noncomputable_draws",
    ],
    "size-extremes": [
        "integration/test_engine.py::test_source_size_disparity_is_visible_without_suppressing_results",
        "contracts/test_memorization_evaluation.py::test_point_scores_match_independently_sampled_training_geometry",
        "test_evaluate_options.py::test_removed_sampling_option_is_rejected",
    ],
    "duplicate-rows": ["test_metamorphic.py::test_duplicate_syn_rows_reduce_coverage"],
    "constant-columns": ["test_fuzz.py::test_constant_columns"],
    "category-alignment": [
        "test_input_contract.py::test_packing_preserves_typed_categories_and_ignores_unused_missing_columns",
        "contracts/runtime/test_category_identity.py::test_equal_values_in_distinct_families_have_distinct_codes",
        "contracts/runtime/test_category_identity.py::test_exact_labels_and_missing_categories_cannot_collide",
        "execution/python/test_primitives.py::test_tv_complement_unseen_level_partial",
    ],
    "temporal-and-unsupported-dtypes": [
        "test_dtypes.py::test_temporal_numeric_declarations_raise_without_converting",
        "test_dtypes.py::test_complex_column_raises",
        "test_dtypes.py::test_nested_object_column_raises",
    ],
    "unicode-categories": ["test_fuzz.py::test_unicode_and_weird_categories"],
    "index-misalignment": ["test_properties.py::test_nondefault_index_does_not_change_results"],
    "target-column": [
        "test_original_mle.py::test_utility_is_optional_registered_and_requires_an_official_target",
        "test_original_mle.py::test_degenerate_classification_targets_are_flagged",
        "test_original_mle.py::test_missing_targets_are_excluded_even_when_predictor_missingness_is_a_category",
        "test_original_mle.py::test_utility_predictor_encoding_never_fits_scoring_rows_or_target",
    ],
    "non-determinism": [
        "integration/test_engine.py::test_determinism_same_seed",
        "test_properties.py::test_full_pipeline_determinism",
    ],
    "multiprocessing": [
        "integration/test_evaluate.py::test_parallel_matches_serial",
        "contracts/runtime/test_execution.py::test_compiled_cpu_preserves_complete_reports_and_every_replicate",
        "contracts/runtime/test_execution.py::test_host_preparation_owns_data_and_preserves_shared_identities",
        "contracts/classifier/test_nonlinear_c2st.py::test_xgboost_worker_budgets_preserve_scores_and_bound_every_fit",
    ],
    "hostile-payloads": [
        "test_fuzz.py::test_all_nan_frame",
        "test_fuzz.py::test_inf_and_denormals",
        "test_fuzz.py::test_unicode_and_weird_categories",
    ],
    "config-dos": [
        "integration/test_evaluate.py::test_automatic_resolution_is_recorded",
        "test_evaluate_options.py::test_removed_sampling_option_is_rejected",
        "contracts/test_fixed_evaluation.py::test_automatic_draw_does_not_truncate_sources_to_a_prefix",
    ],
    "unsupported-execution": [
        "contracts/runtime/test_execution.py::test_execution_options_are_validated_without_torch",
        "contracts/runtime/test_execution.py::test_cpu_fp32_classifier_supported_and_unavailable_cuda_raises",
        "contracts/runtime/test_execution.py::test_explicit_cuda_metric_is_not_silently_run_on_cpu",
    ],
    "tensor-dependency-failure": [
        "contracts/runtime/test_execution.py::test_missing_and_broken_torch_dependencies_are_distinguished",
        "contracts/runtime/test_execution.py::test_custom_execution_does_not_import_torch",
        "contracts/distance/test_torch_distances.py::test_cuda_dependency_errors_and_automatic_cpu_fallback",
    ],
    "workload-routing-mismatch": [
        "contracts/runtime/test_execution.py::test_default_frontend_is_torch_and_small_reports_use_cpu",
        "contracts/runtime/test_execution.py::test_generated_size_drives_default_routing_instead_of_source_rows",
        "contracts/runtime/test_execution.py::test_large_workload_with_tiny_gpu_budget_routes_cpu_before_cuda_probe",
        "contracts/runtime/test_execution.py::test_numeric_dcr_uses_native_cpu_for_auto_and_explicit_options",
    ],
    "tensor-binary-incompatibility": [
        "packaging/test_native_packaging.py::test_incompatible_artifact_is_rejected_before_load",
        "packaging/test_native_packaging.py::test_compatible_binary_is_discovered",
        "packaging/test_native_packaging.py::test_corrupt_binary_is_rejected",
        "packaging/test_native_packaging.py::test_glibc_requirement_is_checked",
        "packaging/test_native_packaging.py::test_release_rpaths_are_relative",
        "packaging/test_native_packaging.py::test_repacked_wheel_has_integral_record_and_exact_torch_dependency",
    ],
    "triton-host-bundle-mismatch": [
        "packaging/test_triton_host.py::test_install_seeds_only_exact_validated_host_entries",
        "packaging/test_triton_host.py::test_incompatible_host_bundle_is_rejected_before_cache_write",
        "packaging/test_triton_host.py::test_corrupt_host_binary_is_rejected",
        "packaging/test_triton_host.py::test_later_corrupt_module_does_not_partially_populate_cache",
        "packaging/test_triton_host.py::test_changing_triton_cache_root_reseeds_host_modules",
        "packaging/test_triton_host.py::test_source_install_does_not_change_triton_builder",
        "packaging/test_triton_host.py::test_launcher_layouts_are_complete_and_independent_of_constexpr_values",
    ],
    "tensor-source-corruption": [
        "contracts/runtime/test_execution.py::test_tensor_storage_precision_ownership_and_gather",
        "contracts/runtime/test_execution.py::test_tensor_gather_rejects_invalid_indices",
        "contracts/runtime/test_execution.py::test_compiled_cpu_preserves_complete_reports_and_every_replicate",
        "contracts/distance/test_torch_distances.py::test_dcr_matches_independent_medians_and_preserves_input_pools",
        "contracts/distance/test_torch_distances.py::test_cpu_cuda_reports_preserve_sampling_and_pairing",
    ],
    "gpu-workspace-overflow": [
        "native/cuda/test_cuda_shape.py::test_shape_workspace_bounds_and_chunk_invariance",
        "contracts/runtime/test_execution.py::test_gpu_budget_bounds_gathers_and_leaves_global_precision_unchanged",
        "contracts/distance/test_torch_distances.py::test_gpu_workspace_peak_and_budget_independence",
        "contracts/fidelity/test_torch_fidelity.py::test_trend_gpu_workspace_multiple_draws_and_sparse_budget_independence",
        "native/cuda/test_cuda_conditional.py::test_workspace_is_bounded_and_draws_are_released",
        "native/cuda/test_cuda_classifier.py::test_scoring_cannot_fit_transforms_and_gpu_workspace_released",
    ],
    "tensor-distance-precision-loss": [
        "contracts/distance/test_torch_distances.py::test_fp64_geometry_never_round_trips_through_fp32",
        "contracts/distance/test_torch_distances.py::test_operator_precision_survives_ambient_autocast",
        "contracts/distance/test_torch_distances.py::test_prdc_matches_independent_onehot_oracle",
        "contracts/distance/test_torch_distances.py::test_dcr_matches_independent_medians_and_preserves_input_pools",
        "contracts/distance/test_torch_distances.py::test_strict_category_boundaries_and_unknown_vs_unknown",
    ],
    "tensor-fidelity-semantic-drift": [
        "native/cuda/test_cuda_shape.py::test_shape_matches_independent_oracle_and_cpu",
        "native/cuda/test_cuda_shape.py::test_shape_complete_reports_precision_and_stream",
        "native/cuda/test_cuda_shape.py::test_shape_sparse_draw_from_actual_large_category_domain",
        "contracts/fidelity/test_torch_fidelity.py::test_trend_matches_dataframe_reference",
        "contracts/fidelity/test_torch_fidelity.py::test_correlations_pairwise_missing_constants_and_high_offsets",
        "contracts/fidelity/test_torch_fidelity.py::test_quantile_edges_ties_and_right_closed_reference_bins",
        "contracts/fidelity/test_torch_fidelity.py::test_contingency_batches_match_independent_occupied_cell_counts",
        "contracts/fidelity/test_torch_fidelity.py::test_actual_none_is_missing_and_literal_nan_is_observed",
        "contracts/fidelity/test_torch_fidelity.py::test_full_trend_reports_preserve_replicates_flags_and_pairing",
        "native/cuda/test_cuda_conditional.py::test_conditional_all_statistics_against_cpu",
        "native/cuda/test_cuda_conditional.py::test_count_and_rarity_boundaries_empty_candidate_and_literal_nan",
        "native/cuda/test_cuda_conditional.py::test_sparse_global_domains_repeated_unequal_draws_and_ideal",
    ],
    "tensor-classifier-semantic-drift": [
        "native/cpu/test_cpu_classifier.py::test_classifier_fold_preprocessing_and_auc_match_python",
        "native/cuda/test_cuda_classifier.py::test_training_fold_preprocessing_and_implicit_adjoint_match_explicit_design",
        "native/cuda/test_cuda_classifier.py::test_wide_design_uses_bounded_implicit_full_one_hot",
        "native/cuda/test_cuda_classifier.py::test_source_fold_conflicts_memory_failures_and_nonconvergence_are_retained",
    ],
    "float-reproducibility": [
        "integration/test_golden.py::test_golden_shape_trend",
        "integration/test_golden.py::test_golden_prdc",
        "test_metamorphic.py::test_affine_scale_invariance_of_prdc",
        "contracts/distance/test_torch_distances.py::test_streaming_radii_keep_duplicate_multiplicities_across_tiles",
    ],
}

_TESTS_DIR = pathlib.Path(__file__).parent
_REF = re.compile(r"^((?:[a-z_]+/)*test_\w+\.py)::(test_\w+)$")
_MODE_ID = re.compile(r'<a id="([a-z0-9]+(?:-[a-z0-9]+)*)"></a> .+')


def _resolves(ref: str, tests_dir: pathlib.Path = _TESTS_DIR) -> bool:
    """True iff *ref* ("file.py::test_name") points at a real test function below the tests root."""
    m = _REF.match(ref)
    if not m:
        return False
    path = tests_dir / m.group(1)
    if not path.is_file():
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == m.group(2)
        for node in ast.parse(path.read_text()).body
    )


def _validate_coverage(catalogue: str, coverage: dict[str, list[str] | str]) -> None:
    modes = []
    for line in catalogue.splitlines():
        if not line.startswith("| ") or line.startswith("| Failure mode |"):
            continue
        cell = line.split("|", 2)[1].strip()
        match = _MODE_ID.fullmatch(cell)
        assert match is not None, f"failure mode missing a stable ID: {cell}"
        modes.append(match.group(1))
    assert modes, "failure-mode catalogue is empty"
    assert len(modes) == len(set(modes)), "duplicate failure-mode IDs"
    missing = set(modes) - coverage.keys()
    stale = coverage.keys() - set(modes)
    assert not missing and not stale, (
        f"catalogue/coverage mismatch: unmapped={sorted(missing)}, stale={sorted(stale)}"
    )
    uncovered = [
        mode for mode, cov in coverage.items() if cov not in (DEFERRED, DOCUMENTED) and not cov
    ]
    assert not uncovered, f"failure modes without a covering test: {uncovered}"


def _validate_references(
    coverage: dict[str, list[str] | str], tests_dir: pathlib.Path = _TESTS_DIR
) -> None:
    unresolved = []
    for mode, cov in coverage.items():
        if cov in (DEFERRED, DOCUMENTED):
            continue
        for ref in cov:
            if not _resolves(ref, tests_dir):
                unresolved.append((mode, ref))
    assert not unresolved, f"coverage refs not resolving to a real test: {unresolved}"


def test_every_failure_mode_has_a_covering_test(checkout_root):
    _validate_coverage((checkout_root / "docs" / "failure-modes.md").read_text(), COVERAGE)


def test_all_coverage_references_resolve_to_real_tests():
    _validate_references(COVERAGE)


@pytest.mark.parametrize(
    "catalogue,coverage,reason",
    [
        ('| <a id="new-mode"></a> New mode | consequence | handling |', {}, "unmapped"),
        (
            '| <a id="new-mode"></a> New mode | consequence | handling |',
            {"old-mode": DOCUMENTED},
            "stale",
        ),
        (
            '| <a id="new-mode"></a> New mode | consequence | handling |',
            {"new-mode": []},
            "without a covering test",
        ),
        ("| New mode | consequence | handling |", {}, "missing a stable ID"),
        ("", {}, "catalogue is empty"),
        (
            '| <a id="same-mode"></a> Mode | x | y |\n| <a id="same-mode"></a> Duplicate | x | y |',
            {"same-mode": DOCUMENTED},
            "duplicate",
        ),
    ],
)
def test_traceability_rejects_incomplete_catalogues(catalogue, coverage, reason):
    with pytest.raises(AssertionError, match=reason):
        _validate_coverage(catalogue, coverage)


@pytest.mark.parametrize(
    "source",
    [
        "def test_renamed(): pass\n",
        "# def test_missing(): pass\n",
        '"""def test_missing(): pass"""\n',
        "def helper():\n    def test_missing(): pass\n",
    ],
)
def test_traceability_rejects_stale_and_non_test_references(tmp_path, source):
    (tmp_path / "test_example.py").write_text(source)
    with pytest.raises(AssertionError, match="test_example.py::test_missing"):
        _validate_references({"mode": ["test_example.py::test_missing"]}, tmp_path)


def test_traceability_accepts_decorated_top_level_tests(tmp_path):
    (tmp_path / "test_example.py").write_text(
        '@pytest.mark.parametrize("value", [1, 2])\ndef test_example(value): pass\n'
    )
    _validate_references({"mode": ["test_example.py::test_example"]}, tmp_path)
