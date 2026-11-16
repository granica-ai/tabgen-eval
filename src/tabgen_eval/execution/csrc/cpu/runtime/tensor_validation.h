#pragma once
#include "cpu/runtime/types.h"
#include <cstdint>
#include <vector>

namespace ltm {
inline void matrix(const Tensor& x, at::ScalarType type) {
  TORCH_CHECK(x.device().is_cpu() && x.dim() == 2 && x.scalar_type() == type && x.is_contiguous(),
              "CPU operators require contiguous matrices with the declared dtype");
}
inline void numbers(const Tensor& x) {
  TORCH_CHECK(x.scalar_type() == at::kDouble || x.scalar_type() == at::kFloat,
              "numeric dtype must be float32 or float64");
  matrix(x, x.scalar_type());
}
/** @brief Validate contiguous CPU numeric [N,PN], int64 [N,PC] and bool [N]
 * buffers; malformed layouts raise TORCH_CHECK before numerical work. */
inline void validate_pool(const Tensor& x, const Tensor& c, const Tensor& complete) {
  numbers(x);
  matrix(c, at::kLong);
  TORCH_CHECK(complete.device().is_cpu() && complete.scalar_type() == at::kBool &&
                  complete.dim() == 1 && complete.is_contiguous(),
              "complete mask must be contiguous CPU bool");
  TORCH_CHECK(x.size(0) == c.size(0) && x.size(0) == complete.numel(), "pool rows must match");
}
/** @brief Validate contiguous CPU int64 [B,Ndraw] indices against the source
 * row count; every consumed index must be nonnegative and in bounds. */
inline void validate_draws(const Tensor& rows, int64_t n) {
  matrix(rows, at::kLong);
  const auto* indices = rows.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < rows.numel(); ++i)
    TORCH_CHECK(indices[i] >= 0 && indices[i] < n, "draw index is outside the input pool");
}

/** @brief Validate category domains only at positions consumed by a draw batch.
 * Pool, draw and cardinality shapes must already have passed validate_pair.
 */
inline void validate_category_draws(const Tensor& categorical, const Tensor& draws,
                                    const std::vector<int64_t>& cards) {
  const auto* codes = categorical.const_data_ptr<int64_t>();
  const auto* rows = draws.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < draws.numel(); ++i)
    for (size_t c = 0; c < cards.size(); ++c) {
      auto code = codes[rows[i] * categorical.size(1) + c];
      TORCH_CHECK(code >= -1 && code < cards[c], "category code outside declared cardinality");
    }
}

void validate_pair(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
                   const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
                   const std::vector<int64_t>& cards);

} // namespace ltm
