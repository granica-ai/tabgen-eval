#pragma once
#include "cpu/runtime/types.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ltm {
/** @brief Bounded column-major reference chunks for 128-lane SIMD operators.
 * Scratch includes the pack and per-worker distance/category accumulators;
 * caller-owned input matrices and output reductions are excluded. */
template <class T> struct ReferenceBlocks {
  int64_t rows, pn, pc, chunk;
  const T* numeric;
  const int64_t* categorical;
  std::vector<T> packed_numeric;
  std::vector<int64_t> packed_codes;
  ReferenceBlocks(const Tensor& x, const Tensor& c, int64_t workers, int64_t scratch_bytes)
      : rows(x.size(0)), pn(x.size(1)), pc(c.size(1)), numeric(x.const_data_ptr<T>()),
        categorical(c.const_data_ptr<int64_t>()) {
    const int64_t state_bytes = workers * 128 * (sizeof(T) + sizeof(int64_t));
    const int64_t row_bytes = pn * sizeof(T) + pc * sizeof(int64_t);
    TORCH_CHECK(scratch_bytes >= state_bytes + std::min<int64_t>(128, rows) * row_bytes,
                "distance scratch budget cannot hold a reference block and worker state");
    chunk = row_bytes ? (scratch_bytes - state_bytes) / row_bytes : rows;
    chunk = std::min<int64_t>({chunk, rows, 16384});
    if (chunk >= 128)
      chunk = (chunk / 128) * 128;
    packed_numeric.resize(pn * chunk);
    packed_codes.resize(pc * chunk);
  }
  int64_t pack(int64_t start) {
    const int64_t count = std::min(chunk, rows - start);
    for (int64_t c = 0; c < pn; ++c)
      for (int64_t j = 0; j < count; ++j)
        packed_numeric[c * chunk + j] = numeric[(start + j) * pn + c];
    for (int64_t c = 0; c < pc; ++c)
      for (int64_t j = 0; j < count; ++j)
        packed_codes[c * chunk + j] = categorical[(start + j) * pc + c];
    return count;
  }
};
} // namespace ltm
