#pragma once
#include "cpu/runtime/thread_budget.h"
#include "cpu/runtime/types.h"
#include <vector>

namespace ltm::preprocessing {
inline int checked_threads(int64_t threads) {
  TORCH_CHECK(threads > 0 && threads <= std::numeric_limits<int>::max(),
              "invalid preprocessing thread budget");
  return int(threads);
}
Tensor count_missing(const Tensor& values, int64_t threads);
Tensor assemble(const Tensor& numbers, const Tensor& codes, int64_t feature_count, bool linear,
                int64_t threads);
Tensor quantize(const Tensor& values, const Tensor& bits, at::ScalarType dtype, int64_t threads);
std::vector<Tensor> fit_numeric(const Tensor& reference, const Tensor& candidate, bool linear,
                                int64_t threads);
Tensor apply_numeric(const Tensor& values, const Tensor& keep, const Tensor& indicators,
                     const Tensor& impute, const Tensor& center, const Tensor& scale, bool linear,
                     int64_t threads);
std::vector<Tensor> fit_categories(const Tensor& reference, const Tensor& candidate,
                                   const Tensor& cardinalities, int64_t threads);
Tensor map_categories(const Tensor& codes, const Tensor& mapping, const Tensor& domains,
                      const Tensor& features, bool linear, int64_t threads);
} // namespace ltm::preprocessing
