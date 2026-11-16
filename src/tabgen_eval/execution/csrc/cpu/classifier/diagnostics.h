#pragma once
#include <cstdint>

// Positional operator diagnostics are a stable wire format. Layout ownership
// lives here, separately from optimizer state and numerical execution.
namespace ltm::classifier_diagnostics {
namespace fold {
enum Field : int64_t {
  training_rows = 0,
  validation_rows = 1,
  encoded_columns = 2,
  converged = 3,
  iterations = 4,
  epochs_completed = 5,
  gradient_squared = 6,
  relative_coefficient_change = 7,
  effective_ridge = 8,
  preconditioner_rank = 9,
  learning_rate = 10,
  maximum_feature_coefficient = 11,
  maximum_feature_change = 12,
  intercept_change = 13,
  stable_inverse = 14,
  estimated_eigenvalue = 15,
  decrement_squared = 16,
  online_gradient_infinity = 17,
  initial_online_gradient_infinity = 18,
  gradient_stopping_threshold = 19,
  accepted_epochs = 20,
  accepted_updates = 21,
  initial_learning_rate = 22,
  final_learning_rate = 23,
  current_attempt_learning_rate = 24,
  backoff_enabled = 25,
  backoff_probe_rows = 26,
  backoff_probe_checks = 27,
  backoff_retries = 28,
  rejected_epochs = 29,
  backoff_retry_limit = 30,
  final_probe_objective = 31,
  schedule_chunk_epochs = 32,
  preparation_training_rows = 33,
  preparation_rows = 34,
  sampled_preparation = 35,
  preparation_sample_hash = 36,
  curvature_data_raw = 37,
  curvature_data_used = 38,
  curvature_row_maximum_raw = 39,
  curvature_row_maximum_used = 40,
  curvature_ridge_bound = 41,
  curvature_practical = 42,
  curvature_calibration_rows = 43,
  curvature_calibration_population = 44,
  curvature_calibration_batch = 45,
  width = 46
};
} // namespace fold
namespace saga_fit {
enum Field : int64_t {
  converged = 0,
  epochs_completed = 1,
  updates = 2,
  gradient_squared = 3,
  relative_coefficient_change = 4,
  learning_rate = 5,
  preconditioner_rank = 6,
  stable_inverse = 7,
  online_gradient_infinity = 8,
  initial_online_gradient_infinity = 9,
  gradient_stopping_threshold = 10,
  accepted_epochs = 11,
  accepted_updates = 12,
  initial_learning_rate = 13,
  final_learning_rate = 14,
  current_attempt_learning_rate = 15,
  backoff_enabled = 16,
  backoff_probe_rows = 17,
  backoff_probe_checks = 18,
  backoff_retries = 19,
  rejected_epochs = 20,
  backoff_retry_limit = 21,
  final_probe_objective = 22,
  schedule_chunk_epochs = 23,
  preparation_training_rows = 24,
  preparation_rows = 25,
  sampled_preparation = 26,
  preparation_sample_hash = 27,
  curvature_data_raw = 28,
  curvature_data_used = 29,
  curvature_row_maximum_raw = 30,
  curvature_row_maximum_used = 31,
  curvature_ridge_bound = 32,
  curvature_practical = 33,
  curvature_calibration_rows = 34,
  curvature_calibration_population = 35,
  curvature_calibration_batch = 36,
  width = 37
};
} // namespace saga_fit
namespace guard {
enum Field : int64_t {
  accepted_epochs = 0,
  accepted_updates = 1,
  initial_learning_rate = 2,
  final_learning_rate = 3,
  current_attempt_learning_rate = 4,
  backoff_enabled = 5,
  backoff_probe_rows = 6,
  backoff_probe_checks = 7,
  backoff_retries = 8,
  rejected_epochs = 9,
  backoff_retry_limit = 10,
  final_probe_objective = 11,
  schedule_chunk_epochs = 12,
  width = 13
};
} // namespace guard
namespace preparation {
enum Field : int64_t {
  preparation_training_rows = 0,
  preparation_rows = 1,
  sampled_preparation = 2,
  preparation_sample_hash = 3,
  curvature_data_raw = 4,
  curvature_data_used = 5,
  curvature_row_maximum_raw = 6,
  curvature_row_maximum_used = 7,
  curvature_ridge_bound = 8,
  curvature_practical = 9,
  curvature_calibration_rows = 10,
  curvature_calibration_population = 11,
  curvature_calibration_batch = 12,
  width = 13
};
} // namespace preparation
} // namespace ltm::classifier_diagnostics
