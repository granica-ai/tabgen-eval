#pragma once
#include "cpu/runtime/types.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace ltm {
/** @brief Optional draw-owned global Trend observations in canonical pair order. */
template <class T> struct FidelityScoreDetail {
  std::vector<std::pair<int64_t, int64_t>> pairs;
  std::vector<T> scores;
};

/** @brief One reference-observed conditioning slice and its existing scores. */
template <class T> struct ConditionalSliceDetail {
  int64_t label = 0, reference_count = 0, candidate_count = 0;
  int64_t reference_total = 0, candidate_total = 0;
  bool eligible = false, rare = false;
  T shape = T(nan), trend = T(nan), recovery = T(nan);
  T lower = T(nan), upper = T(nan);
  bool left_closed = false, right_closed = false;
};

/** @brief Per-column collectors; parallel conditioning workers own distinct rows. */
template <class T> struct ConditionalDetail {
  std::vector<std::vector<T>> columns;
  std::vector<std::vector<ConditionalSliceDetail<T>>> slices;
};
} // namespace ltm
