#pragma once
#include "cpu/runtime/types.h"

namespace ltm {
/** @brief Inclusive self-excluded neighborhood counts for multiple squared radii.
 * Counts [R,N] retain every duplicate row. Categories use global identities and
 * contribute exactly two for each unequal column; no OOV remapping occurs. */
Tensor local_concentration_counts(const Tensor& numeric, const Tensor& categorical,
                                  const Tensor& squared_thresholds, int64_t workers,
                                  int64_t scratch_bytes);
/** @brief Direct squared distances for aligned indexed record pairs. */
Tensor paired_concentration_distances(const Tensor& numeric, const Tensor& categorical,
                                      const Tensor& left, const Tensor& right, int64_t workers);
} // namespace ltm
