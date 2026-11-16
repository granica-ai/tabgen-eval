#pragma once
#include "cpu/runtime/reductions.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ltm {
template <class T> std::vector<T> sorted(const std::vector<T>& values) {
  std::vector<T> v;
  v.reserve(values.size());
  for (auto x : values)
    if (std::isfinite(x))
      v.push_back(x);
  std::sort(v.begin(), v.end());
  return v;
}
/** @brief Return the empirical KS complement for sorted finite observations;
 * ties advance together and an empty distribution returns NaN. */
template <class T> T ks(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.empty() || b.empty())
    return T(nan);
  size_t i = 0, j = 0;
  T gap = 0;
  while (i < a.size() || j < b.size()) {
    T x = j == b.size() || (i < a.size() && a[i] < b[j]) ? a[i] : b[j];
    while (i < a.size() && a[i] <= x)
      ++i;
    while (j < b.size() && b[j] <= x)
      ++j;
    gap = std::max(gap, std::abs(T(i) / T(a.size()) - T(j) / T(b.size())));
  }
  return T(1) - gap;
}
template <class T>
std::vector<T> edges_sorted(const std::vector<T>& v, int64_t bins, bool quantile) {
  if (v.empty())
    return {};
  T lo = v.front(), hi = v.back();
  std::vector<T> e;
  if (lo == hi) {
    T pad = std::abs(lo) * T(1e-6) + T(1e-9);
    lo -= pad;
    hi += pad;
    quantile = false;
  }
  for (int64_t i = 0; i <= bins; ++i) {
    T z;
    if (quantile) {
      T q = i == bins ? T(1) : T(i) * T(1.0 / bins);
      T p = q * T(v.size() - 1);
      auto low = std::min<size_t>(std::floor(p), v.size() - 1),
           high = std::min(low + 1, v.size() - 1);
      T f = p - T(low), d = v[high] - v[low];
      z = f >= T(.5) ? v[high] - d * (T(1) - f) : v[low] + d * f;
    } else
      z = i == bins ? hi : lo + T(i) * ((hi - lo) / T(bins));
    if (e.empty() || z != e.back())
      e.push_back(z);
  }
  if (e.size() < 2)
    e.clear();
  return e;
}
template <class T> std::vector<T> edges(const std::vector<T>& values, int64_t bins, bool quantile) {
  return edges_sorted(sorted(values), bins, quantile);
}
template <class T>
std::vector<int64_t> bin(const std::vector<T>& x, const std::vector<T>& e, bool clip,
                         bool infinite) {
  std::vector<int64_t> out(x.size(), -1);
  if (e.size() < 2)
    return out;
  for (size_t i = 0; i < x.size(); ++i) {
    T v = x[i];
    if (std::isnan(v) || (!infinite && !std::isfinite(v)))
      continue;
    if (clip)
      v = std::min(std::max(v, e.front()), e.back());
    if (!std::isfinite(v) || v < e.front() || v > e.back())
      continue;
    out[i] = v == e.front() ? 0 : std::lower_bound(e.begin(), e.end(), v) - e.begin() - 1;
  }
  return out;
}
// Shift and normalize in the selected precision before centering. Scaling
// prevents both overflow and underflow without introducing wider arithmetic.
template <class T> std::vector<T> centered(std::vector<T> values) {
  if (values.empty())
    return values;
  T anchor = values[0], scale = 0;
  for (auto value : values)
    scale = std::max(scale, std::abs(value - anchor));
  if (std::isfinite(scale)) {
    for (auto& value : values)
      value = scale > 0 ? (value - anchor) / scale : T(0);
  } else {
    T absolute = 0;
    for (auto value : values)
      absolute = std::max(absolute, std::abs(value));
    for (auto& value : values)
      value = value / absolute - anchor / absolute;
  }
  T center = mean(values);
  for (auto& value : values)
    value -= center;
  return values;
}
template <class T> T pearson(const std::vector<T>& x, const std::vector<T>& y) {
  std::vector<T> a, b;
  for (size_t i = 0; i < x.size(); ++i)
    if (std::isfinite(x[i]) && std::isfinite(y[i])) {
      a.push_back(x[i]);
      b.push_back(y[i]);
    }
  if (a.size() < 2)
    return T(nan);
  a = centered(std::move(a));
  b = centered(std::move(b));
  T xx = 0, yy = 0, xy = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    xx += a[i] * a[i];
    yy += b[i] * b[i];
    xy += a[i] * b[i];
  }
  return xx > 0 && yy > 0 ? std::clamp(xy / (std::sqrt(xx) * std::sqrt(yy)), T(-1), T(1)) : T(nan);
}
struct PairHash {
  size_t operator()(const std::pair<int64_t, int64_t>& p) const {
    uint64_t a = p.first, b = p.second;
    return (a * 0x9e3779b97f4a7c15ULL) ^ (b + 0x517cc1b727220a95ULL + (a << 6) + (a >> 2));
  }
};
/** @brief Reduce exact occupied joint counts in selected precision; dense
 * scratch is bounded at 4096 cells, wider domains use sorted occupied tuple
 * keys. */
template <class T>
T contingency(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
              const std::vector<int64_t>& c, const std::vector<int64_t>& d, int64_t na,
              int64_t nb) {
  if (!na || !nb)
    return T(nan);
  int64_t nr = 0, ns = 0;
  T delta = 0, correction = 0;
  auto accumulate = [&](int64_t r, int64_t s) {
    T value = std::abs(T(r) / T(nr) - T(s) / T(ns));
    T y = value - correction, next = delta + y;
    correction = (next - delta) - y;
    delta = next;
  };
  if (na <= 4096 && nb <= 4096 && na * nb <= 4096) {
    // One bounded buffer per worker, reused across millions of slice/pair
    // evaluations. Integer counts stay exact and reduction order stays fixed.
    thread_local std::vector<std::pair<int64_t, int64_t>> counts;
    counts.assign(na * nb, {0, 0});
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] >= 0 && b[i] >= 0) {
        ++counts[a[i] * nb + b[i]].first;
        ++nr;
      }
    for (size_t i = 0; i < c.size(); ++i)
      if (c[i] >= 0 && d[i] >= 0) {
        ++counts[c[i] * nb + d[i]].second;
        ++ns;
      }
    if (!nr || !ns)
      return T(nan);
    for (auto p : counts)
      accumulate(p.first, p.second);
  } else {
    std::unordered_map<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>, PairHash> counts;
    counts.reserve(a.size() + c.size());
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] >= 0 && b[i] >= 0) {
        ++counts[{a[i], b[i]}].first;
        ++nr;
      }
    for (size_t i = 0; i < c.size(); ++i)
      if (c[i] >= 0 && d[i] >= 0) {
        ++counts[{c[i], d[i]}].second;
        ++ns;
      }
    if (!nr || !ns)
      return T(nan);
    std::vector<std::pair<int64_t, int64_t>> keys;
    keys.reserve(counts.size());
    for (auto& kv : counts)
      keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (auto key : keys) {
      auto p = counts[key];
      accumulate(p.first, p.second);
    }
  }
  return std::clamp(T(1) - T(.5) * delta, T(0), T(1));
}
// Marginal counts need one code per observation, with no artificial second
// dimension or draw-sized scratch. Accessors also let Shape read directly
// from its source columns instead of materializing the sampled table.
/** @brief Compute the marginal TV complement using exact identity counts;
 * negative codes are excluded and empty marginals return NaN. */
template <class T, class A, class B> T marginal_tv(A a, size_t na, B b, size_t nb, int64_t levels) {
  int64_t nr = 0, ns = 0;
  T delta = 0, correction = 0;
  auto accumulate = [&](int64_t r, int64_t s) {
    T value = std::abs(T(r) / T(nr) - T(s) / T(ns));
    T y = value - correction, next = delta + y;
    correction = (next - delta) - y;
    delta = next;
  };
  if (levels <= 4096) {
    thread_local std::vector<std::pair<int64_t, int64_t>> counts;
    counts.assign(levels, {0, 0});
    for (size_t i = 0; i < na; ++i) {
      auto code = a(i);
      if (code >= 0) {
        ++counts[code].first;
        ++nr;
      }
    }
    for (size_t i = 0; i < nb; ++i) {
      auto code = b(i);
      if (code >= 0) {
        ++counts[code].second;
        ++ns;
      }
    }
    if (!nr || !ns)
      return T(nan);
    for (auto count : counts)
      accumulate(count.first, count.second);
  } else {
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> counts;
    counts.reserve(std::min<uint64_t>(levels, uint64_t(na) + uint64_t(nb)));
    for (size_t i = 0; i < na; ++i) {
      auto code = a(i);
      if (code >= 0) {
        ++counts[code].first;
        ++nr;
      }
    }
    for (size_t i = 0; i < nb; ++i) {
      auto code = b(i);
      if (code >= 0) {
        ++counts[code].second;
        ++ns;
      }
    }
    if (!nr || !ns)
      return T(nan);
    std::vector<int64_t> keys;
    keys.reserve(counts.size());
    for (auto& count : counts)
      keys.push_back(count.first);
    std::sort(keys.begin(), keys.end());
    for (auto key : keys) {
      auto count = counts[key];
      accumulate(count.first, count.second);
    }
  }
  return std::clamp(T(1) - T(.5) * delta, T(0), T(1));
}
template <class T>
T tv(const std::vector<int64_t>& a, const std::vector<int64_t>& b, int64_t levels) {
  return marginal_tv<T>([&](size_t i) { return a[i]; }, a.size(), [&](size_t i) { return b[i]; },
                        b.size(), levels);
}

} // namespace ltm
