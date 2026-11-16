#pragma once

#include <ATen/ATen.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ltm::cuda::classifier {
// Wire ordering stays stable at the Torch operator boundary. Decode once here;
// numerical implementations use names instead of positional tensor indices.
struct FitPacketView {
  const at::Tensor& numbers;
  const at::Tensor& categories;
  const at::Tensor& category_centers;
  const at::Tensor& category_scales;
  const at::Tensor& labels;
  const at::Tensor& schedule;
  const at::Tensor& inverse_root;
  const at::Tensor& reflectors;
  const at::Tensor& tau;
  const at::Tensor& rotations;
  const at::Tensor& ratios;
  const at::Tensor& basis;
  const at::Tensor& initial_weights;
  const at::Tensor& initial_table;
  const at::Tensor& initial_average;
  explicit FitPacketView(const std::vector<at::Tensor>& values)
      : numbers(entry(values, 0)), categories(entry(values, 1)), category_centers(entry(values, 2)),
        category_scales(entry(values, 3)), labels(entry(values, 4)), schedule(entry(values, 5)),
        inverse_root(entry(values, 6)), reflectors(entry(values, 7)), tau(entry(values, 8)),
        rotations(entry(values, 9)), ratios(entry(values, 10)), basis(entry(values, 11)),
        initial_weights(entry(values, 12)), initial_table(entry(values, 13)),
        initial_average(entry(values, 14)) {}

private:
  static const at::Tensor& entry(const std::vector<at::Tensor>& values, size_t index) {
    TORCH_CHECK(values.size() == 15, "native SAGA tensor packet must contain15 entries");
    return values[index];
  }
};
struct PreparationPacket {
  at::Tensor numbers;
  at::Tensor categories;
  at::Tensor category_centers;
  at::Tensor category_scales;
  at::Tensor category_rows;
  at::Tensor category_offsets;
  explicit PreparationPacket(const std::vector<at::Tensor>& values)
      : numbers(entry(values, 0)), categories(entry(values, 1)), category_centers(entry(values, 2)),
        category_scales(entry(values, 3)), category_rows(entry(values, 4)),
        category_offsets(entry(values, 5)) {}

private:
  static const at::Tensor& entry(const std::vector<at::Tensor>& values, size_t index) {
    TORCH_CHECK(values.size() == 6, "native preparation packet needs six tensors");
    return values[index];
  }
};
struct FactorPacket {
  at::Tensor inverse_root;
  at::Tensor reflectors;
  at::Tensor tau;
  at::Tensor rotations;
  at::Tensor ratios;
  at::Tensor basis;
  at::Tensor residual;
  at::Tensor fallback;
  explicit FactorPacket(const std::vector<at::Tensor>& values)
      : inverse_root(entry(values, 0)), reflectors(entry(values, 1)), tau(entry(values, 2)),
        rotations(entry(values, 3)), ratios(entry(values, 4)), basis(entry(values, 5)),
        residual(entry(values, 6)), fallback(entry(values, 7)) {}

private:
  static const at::Tensor& entry(const std::vector<at::Tensor>& values, size_t index) {
    TORCH_CHECK(values.size() == 8, "native factor result must contain eight tensors");
    return values[index];
  }
};
namespace epoch_history {
constexpr int maximum_abs_coefficient = 0;
constexpr int maximum_abs_change = 1;
constexpr int finite = 2;
constexpr int intercept_abs_change = 3;
constexpr int online_gradient_infinity = 4;
constexpr int initial_gradient_infinity = 5;
constexpr int width = 6;
} // namespace epoch_history
struct PreparationDiagnostics {
  double fallback;
  double data_curvature;
  double learning_rate;
  double residual_diagonal_min;
  double inverse_ratio_min;
  double full_rows;
  double curvature_rows;
  double sampled;
  double sample_hash;
  double data_curvature_raw;
  double data_curvature_used;
  double row_maximum_raw;
  double row_maximum_used;
  double ridge_bound;
  double practical_curvature;
  double calibration_rows;
  double calibration_population;
  double calibration_batch;
  at::Tensor tensor() const {
    return at::tensor({fallback, data_curvature, learning_rate, residual_diagonal_min,
                       inverse_ratio_min, full_rows, curvature_rows, sampled, sample_hash,
                       data_curvature_raw, data_curvature_used, row_maximum_raw, row_maximum_used,
                       ridge_bound, practical_curvature, calibration_rows, calibration_population,
                       calibration_batch},
                      at::TensorOptions().dtype(at::kDouble));
  }
};
} // namespace ltm::cuda::classifier
