#pragma once
#include "cpu/runtime/types.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace ltm {
template <class T> T mean(const std::vector<T>& a) {
  T sum = 0, correction = 0;
  int64_t n = 0;
  for (auto v : a)
    if (std::isfinite(v)) {
      T y = v - correction, t = sum + y;
      correction = (t - sum) - y;
      sum = t;
      ++n;
    }
  return n ? sum / T(n) : T(nan);
}

} // namespace ltm
