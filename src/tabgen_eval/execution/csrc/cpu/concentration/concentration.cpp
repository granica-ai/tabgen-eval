#include "cpu/concentration/concentration.h"
#include "cpu/distance/blocks.h"
#include "cpu/distance/packing.h"
#include "cpu/runtime/tensor_validation.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ltm {
namespace {
template <class T> void finite_numeric(const Tensor& x) {
  auto* values = x.const_data_ptr<T>();
  TORCH_CHECK(std::all_of(values, values + x.numel(), [](T v) { return std::isfinite(v); }),
              "concentration requires finite numeric coordinates");
}

template <class T>
Tensor counts_impl(const Tensor& x, const Tensor& codes, const Tensor& thresholds, int64_t threads,
                   int64_t scratch_bytes) {
  const auto n = x.size(0), pn = x.size(1), pc = codes.size(1), nr = thresholds.numel();
  auto output = at::zeros({nr, n}, codes.options());
  if (!n || !nr)
    return output;
  finite_numeric<T>(x);
  const auto* cutoffs = thresholds.const_data_ptr<T>();
  TORCH_CHECK(std::all_of(cutoffs, cutoffs + nr, [](T v) { return !std::isnan(v) && v >= T(0); }),
              "squared concentration thresholds must be nonnegative and not NaN");
  const auto* values = x.const_data_ptr<T>();
  const auto* categories = codes.const_data_ptr<int64_t>();
  auto* counts = output.data_ptr<int64_t>();
  const auto workers = std::min(threads, n);
  ReferenceBlocks<T> blocks(x, codes, workers, scratch_bytes);
  for (int64_t start = 0; start < n; start += blocks.chunk) {
    const int64_t count = blocks.pack(start);
#pragma omp parallel for num_threads(workers) if (workers > 1)
    for (int64_t i = 0; i < n; ++i) {
      alignas(64) T distances[128];
      for (int64_t begin = 0; begin < count; begin += 128) {
        auto length = std::min<int64_t>(128, count - begin);
        distance_block<T, false>(
            pn, pc, [&](int64_t c) { return values[i * pn + c]; },
            [&](int64_t c) { return categories[i * pc + c]; },
            [&](int64_t c) { return blocks.packed_numeric.data() + c * blocks.chunk + begin; },
            [&](int64_t c) { return blocks.packed_codes.data() + c * blocks.chunk + begin; },
            length, distances);
        if (start + begin <= i && i < start + begin + length)
          distances[i - start - begin] = std::numeric_limits<T>::infinity();
        for (int64_t radius = 0; radius < nr; ++radius) {
          int64_t hits = 0;
          const T threshold = cutoffs[radius];
#pragma omp simd reduction(+ : hits)
          for (int64_t j = 0; j < length; ++j)
            hits += start + begin + j != i && distances[j] <= threshold;
          counts[radius * n + i] += hits;
        }
      }
    }
  }
  return output;
}

template <class T>
Tensor pairs_impl(const Tensor& x, const Tensor& codes, const Tensor& left, const Tensor& right,
                  int64_t workers) {
  finite_numeric<T>(x);
  const auto n = x.size(0), pn = x.size(1), pc = codes.size(1), k = left.numel();
  const auto* values = x.const_data_ptr<T>();
  const auto* categories = codes.const_data_ptr<int64_t>();
  const auto* a = left.const_data_ptr<int64_t>();
  const auto* b = right.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < k; ++i)
    TORCH_CHECK(a[i] >= 0 && a[i] < n && b[i] >= 0 && b[i] < n,
                "concentration pair index is outside the input pool");
  auto output = at::empty({k}, x.options());
  auto* out = output.data_ptr<T>();
#pragma omp parallel for num_threads(workers) if (workers > 1)
  for (int64_t i = 0; i < k; ++i) {
    T distance = 0;
    int64_t category = 0;
    for (int64_t c = 0; c < pn; ++c) {
      T delta = values[a[i] * pn + c] - values[b[i] * pn + c];
      distance += delta * delta;
    }
    for (int64_t c = 0; c < pc; ++c)
      category += 2 * (categories[a[i] * pc + c] != categories[b[i] * pc + c]);
    out[i] = distance + T(category);
  }
  return output;
}
} // namespace

Tensor local_concentration_counts(const Tensor& x, const Tensor& codes, const Tensor& thresholds,
                                  int64_t workers, int64_t scratch_bytes) {
  numbers(x);
  matrix(codes, at::kLong);
  TORCH_CHECK(x.size(0) == codes.size(0) && thresholds.device().is_cpu() && thresholds.dim() == 1 &&
                  thresholds.is_contiguous() && thresholds.scalar_type() == x.scalar_type(),
              "incompatible concentration feature/threshold layouts");
  TORCH_CHECK(workers > 0 && scratch_bytes >= 0,
              "concentration requires positive workers and nonnegative scratch bytes");
  return x.scalar_type() == at::kDouble
             ? counts_impl<double>(x, codes, thresholds, workers, scratch_bytes)
             : counts_impl<float>(x, codes, thresholds, workers, scratch_bytes);
}

Tensor paired_concentration_distances(const Tensor& x, const Tensor& codes, const Tensor& left,
                                      const Tensor& right, int64_t workers) {
  numbers(x);
  matrix(codes, at::kLong);
  TORCH_CHECK(x.size(0) == codes.size(0) && workers > 0, "invalid concentration rows/workers");
  for (const auto& indices : {left, right})
    TORCH_CHECK(indices.device().is_cpu() && indices.dim() == 1 && indices.is_contiguous() &&
                    indices.scalar_type() == at::kLong,
                "concentration pair indices require contiguous CPU int64 vectors");
  TORCH_CHECK(left.numel() == right.numel(), "concentration pair index lengths differ");
  return x.scalar_type() == at::kDouble ? pairs_impl<double>(x, codes, left, right, workers)
                                        : pairs_impl<float>(x, codes, left, right, workers);
}
} // namespace ltm
