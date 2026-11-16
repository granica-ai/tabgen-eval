#pragma once
#include "cpu/fidelity/statistics.h"
#include "cpu/runtime/types.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ltm {
template <class T>
T shape(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc, const int64_t* rd,
        int64_t nr, const int64_t* sd, int64_t ns, const std::vector<int64_t>& cards, int threads,
        std::vector<T>* details = nullptr) {
  const auto *r = rn.const_data_ptr<T>(), *s = sn.const_data_ptr<T>();
  const auto *cr = rc.const_data_ptr<int64_t>(), *cs = sc.const_data_ptr<int64_t>();
  int64_t numeric = rn.size(1), categorical = rc.size(1);
  std::vector<T> scores(numeric + categorical);
  // Cyclic assignment spreads expensive numeric sorts over every worker even
  // when cheap categorical columns follow the entire numeric block.
#pragma omp parallel for schedule(static, 1)                                                       \
    num_threads(threads) if (threads > 1 && scores.size() > 4)
  for (int64_t c = 0; c < (int64_t)scores.size(); ++c) {
    if (c < numeric) {
      std::vector<T> a, b;
      a.reserve(nr);
      b.reserve(ns);
      for (int64_t i = 0; i < nr; ++i) {
        T value = r[rd[i] * numeric + c];
        if (std::isfinite(value))
          a.push_back(value);
      }
      for (int64_t i = 0; i < ns; ++i) {
        T value = s[sd[i] * numeric + c];
        if (std::isfinite(value))
          b.push_back(value);
      }
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      scores[c] = ks(a, b);
    } else {
      int64_t j = c - numeric;
      scores[c] =
          marginal_tv<T>([&](size_t i) { return cr[rd[i] * categorical + j]; }, nr,
                         [&](size_t i) { return cs[sd[i] * categorical + j]; }, ns, cards[j]);
    }
  }
  T result = mean(scores);
  if (details)
    *details = std::move(scores);
  return result;
}

} // namespace ltm
