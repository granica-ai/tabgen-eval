#pragma once
#include <algorithm>
#include <cstdint>

namespace ltm {
/** @brief SIMD distances from one query to at most 128 reference rows.
 * Accessors supply query scalars and contiguous column-major reference slices.
 * Numeric accumulation preserves feature order and category counts remain exact. */
template <class T, bool unknown_zero = true, class QueryNumeric, class QueryCategory,
          class ReferenceNumeric, class ReferenceCategory>
void distance_block(int64_t pn, int64_t pc, QueryNumeric query_numeric,
                    QueryCategory query_category, ReferenceNumeric reference_numeric,
                    ReferenceCategory reference_category, int64_t length, T* out) {
  alignas(64) int64_t categories[128]{};
  std::fill(out, out + length, T(0));
  for (int64_t c = 0; c < pn; ++c) {
    T x = query_numeric(c);
    const T* y = reference_numeric(c);
#pragma omp simd
    for (int64_t j = 0; j < length; ++j) {
      T delta = x - y[j];
      out[j] += delta * delta;
    }
  }
  for (int64_t c = 0; c < pc; ++c) {
    int64_t x = query_category(c);
    const int64_t* y = reference_category(c);
#pragma omp simd
    for (int64_t j = 0; j < length; ++j) {
      if constexpr (unknown_zero)
        categories[j] += 2 * (x != y[j]) - ((x < 0) != (y[j] < 0));
      else
        categories[j] += 2 * (x != y[j]);
    }
  }
#pragma omp simd
  for (int64_t j = 0; j < length; ++j)
    out[j] += T(categories[j]);
}
} // namespace ltm
