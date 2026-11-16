#pragma once
#include "cpu/fidelity/details.h"
#include "cpu/runtime/sample.h"
#include <vector>

namespace ltm {
/** @brief Return selected-precision Shape/Trend scores for draw-owned columns,
 * using per-pair missingness and reference-fitted bins. Inputs remain
 * read-only. */
template <class T>
std::vector<T> fidelity(const Sample<T>& a, const Sample<T>& b, const std::vector<int64_t>& levels,
                        bool global, int64_t bins, int threads,
                        FidelityScoreDetail<T>* details = nullptr, int64_t components = 3);
/** @brief Return six unweighted Conditional statistics over retained reference
 * slices; group recovery and rarity preserve exact integer counts. */
template <class T>
std::vector<T> conditional(const Sample<T>& a, const Sample<T>& b,
                           const std::vector<int64_t>& levels, int64_t count, int64_t bins,
                           double rare, int threads, ConditionalDetail<T>* details = nullptr,
                           int64_t components = 7);
/** @brief Test applicability of rare-only Conditional output using the same
 * retained slice/count rules as the numerical statistic. */
template <class T>
bool has_rare_sample(const Sample<T>& a, int64_t count, int64_t bins, double rare);

} // namespace ltm
