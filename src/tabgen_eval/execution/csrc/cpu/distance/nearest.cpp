#include "cpu/distance/nearest.h"
#include "cpu/distance/blocks.h"
#include "cpu/distance/packing.h"
#include "cpu/runtime/tensor_validation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ltm {
namespace {
template <class T>
Tensor nearest_impl(const Tensor& reference, const Tensor& query, const Tensor& rc,
                    const Tensor& qc, bool exclude_self, int64_t threads, int64_t scratch_bytes) {
  const auto nr = reference.size(0), nq = query.size(0);
  const auto pn = reference.size(1), pc = rc.size(1);
  const auto* r = reference.const_data_ptr<T>();
  const auto* q = query.const_data_ptr<T>();
  const auto* cq = qc.const_data_ptr<int64_t>();
  Tensor output = at::full({nq}, std::numeric_limits<T>::infinity(), query.options());
  if (!nq || !nr)
    return output;
  TORCH_CHECK(std::all_of(r, r + reference.numel(), [](T v) { return std::isfinite(v); }) &&
                  std::all_of(q, q + query.numel(), [](T v) { return std::isfinite(v); }),
              "nearest distances require finite numeric coordinates");
  const int64_t workers = std::min(threads, nq);
  ReferenceBlocks<T> blocks(reference, rc, workers, scratch_bytes);
  auto* result = output.data_ptr<T>();
  for (int64_t start = 0; start < nr; start += blocks.chunk) {
    const int64_t count = blocks.pack(start);
#pragma omp parallel for num_threads(workers) if (workers > 1)
    for (int64_t i = 0; i < nq; ++i) {
      T minimum = result[i];
      if (minimum == T(0))
        continue;
      alignas(64) T values[128];
      for (int64_t begin = 0; begin < count; begin += 128) {
        auto length = std::min<int64_t>(128, count - begin);
        distance_block<T>(
            pn, pc, [&](int64_t c) { return q[i * pn + c]; },
            [&](int64_t c) { return cq[i * pc + c]; },
            [&](int64_t c) { return blocks.packed_numeric.data() + c * blocks.chunk + begin; },
            [&](int64_t c) { return blocks.packed_codes.data() + c * blocks.chunk + begin; },
            length, values);
        if (exclude_self && start + begin <= i && i < start + begin + length)
          values[i - start - begin] = std::numeric_limits<T>::infinity();
        for (int64_t j = 0; j < length; ++j)
          minimum = std::min(minimum, values[j]);
        if (minimum == T(0))
          break;
      }
      result[i] = minimum;
    }
  }
#pragma omp parallel for num_threads(workers) if (workers > 1)
  for (int64_t i = 0; i < nq; ++i)
    result[i] = std::sqrt(result[i]);
  return output;
}
} // namespace

Tensor nearest_distances(const Tensor& reference, const Tensor& query, const Tensor& rc,
                         const Tensor& qc, bool exclude_self, int64_t threads,
                         int64_t scratch_bytes) {
  numbers(reference);
  numbers(query);
  matrix(rc, at::kLong);
  matrix(qc, at::kLong);
  TORCH_CHECK(reference.scalar_type() == query.scalar_type() &&
                  reference.size(1) == query.size(1) && rc.size(1) == qc.size(1) &&
                  reference.size(0) == rc.size(0) && query.size(0) == qc.size(0),
              "incompatible nearest distance layouts");
  TORCH_CHECK(!exclude_self || reference.size(0) == query.size(0),
              "positional self exclusion requires equal row counts");
  TORCH_CHECK(threads > 0 && scratch_bytes >= 0,
              "nearest distances require positive threads and nonnegative scratch bytes");
  return reference.scalar_type() == at::kDouble
             ? nearest_impl<double>(reference, query, rc, qc, exclude_self, threads, scratch_bytes)
             : nearest_impl<float>(reference, query, rc, qc, exclude_self, threads, scratch_bytes);
}
} // namespace ltm
