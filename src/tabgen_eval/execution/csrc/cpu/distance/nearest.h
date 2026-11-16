#pragma once
#include "cpu/runtime/types.h"

namespace ltm {
/** @brief Euclidean nearest distances in an already-fitted compact geometry.
 * A bounded column-major reference pack enables SIMD without full-table copies.
 * Self exclusion masks only identical row positions, retaining other duplicates. */
Tensor nearest_distances(const Tensor& reference, const Tensor& query,
                         const Tensor& reference_codes, const Tensor& query_codes,
                         bool exclude_self, int64_t threads, int64_t scratch_bytes);
} // namespace ltm
