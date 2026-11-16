#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace ltm::preprocessing {
// Missing (-1) is an ordinary, distinct level when observed during training.
// No held-out observations enter this union.
template <class Getter>
std::vector<int64_t> active_categories(int64_t nr, int64_t ns, Getter code) {
  std::unordered_set<int64_t> active;
  for (int g = 0; g < 2; ++g)
    for (int64_t r = 0; r < (g ? ns : nr); ++r)
      active.insert(code(g, r));
  std::vector<int64_t> result(active.begin(), active.end());
  std::sort(result.begin(), result.end());
  return result;
}
} // namespace ltm::preprocessing
