/** @file
 * @brief Reference-fitted compact distance search with bounded pairwise
 * storage.
 */
#include "cpu/distance/distance.h"
#include "cpu/distance/blocks.h"
#include "cpu/runtime/reductions.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <unordered_set>
namespace ltm {
template <class T> struct Geometry {
  std::vector<T> mins, ranges;
  std::vector<std::unordered_set<int64_t>> levels;
  explicit Geometry(const Sample<T>& a) {
    for (auto& col : a.numeric) {
      T lo = std::numeric_limits<T>::infinity(), hi = -lo;
      for (int64_t i = 0; i < a.n; ++i)
        if (a.complete[i]) {
          lo = std::min(lo, col[i]);
          hi = std::max(hi, col[i]);
        }
      mins.push_back(lo);
      ranges.push_back(lo == hi ? T(1) : hi - lo);
    }
    for (auto& col : a.categorical) {
      std::unordered_set<int64_t> set;
      for (int64_t i = 0; i < a.n; ++i)
        if (a.complete[i])
          set.insert(col[i]);
      levels.push_back(std::move(set));
    }
  }
  Sample<T> transform(const Sample<T>& a) {
    std::vector<int64_t> rows;
    for (int64_t i = 0; i < a.n; ++i)
      if (a.complete[i])
        rows.push_back(i);
    auto s = select(a, rows);
    for (size_t c = 0; c < s.numeric.size(); ++c)
      for (auto& x : s.numeric[c])
        x = (x - mins[c]) / ranges[c];
    for (size_t c = 0; c < s.categorical.size(); ++c)
      for (auto& v : s.categorical[c])
        if (v < 0 || !levels[c].count(v))
          v = -1;
    return s;
  }
};
template <class T> bool finite(const Sample<T>& s) {
  for (auto& c : s.numeric)
    for (auto v : c)
      if (!std::isfinite(v))
        return false;
  return true;
}
// Vectorize across independent reference rows, preserving the feature-order
// FP32/FP64 accumulation and integer category contribution for each distance.
template <class T>
void block(const Sample<T>& a, int64_t q, const Sample<T>& b, int64_t begin, int64_t length,
           T* out) {
  distance_block<T>(
      a.numeric.size(), a.categorical.size(), [&](int64_t c) { return a.numeric[c][q]; },
      [&](int64_t c) { return a.categorical[c][q]; },
      [&](int64_t c) { return b.numeric[c].data() + begin; },
      [&](int64_t c) { return b.categorical[c].data() + begin; }, length, out);
}
template <class T> std::vector<T> radii(const Sample<T>& a, int64_t k, int threads) {
  std::vector<T> out(a.n);
  int workers = std::max<int64_t>(
      1, std::min<int64_t>(threads, (256LL * 1024 * 1024) / std::max<int64_t>(1, a.n * sizeof(T))));
#pragma omp parallel num_threads(workers) if (workers > 1)
  {
    std::vector<T> distances(a.n);
#pragma omp for
    for (int64_t i = 0; i < a.n; ++i) {
      for (int64_t r = 0; r < a.n; r += 128)
        block(a, i, a, r, std::min<int64_t>(128, a.n - r), distances.data() + r);
      std::nth_element(distances.begin(), distances.begin() + k, distances.end());
      out[i] = distances[k];
    }
  }
  return out;
}
template <class T>
std::vector<double> prdc(const Sample<T>& a, const Sample<T>& b, int64_t k, int threads,
                         int64_t components) {
  Geometry<T> geometry(a);
  auto r = geometry.transform(a), s = geometry.transform(b);
  bool wants_real = components & 13, wants_recall = components & 2;
  bool wants_counts = components & 5, wants_cover = components & 8;
  if (!components || !r.n || !s.n || (wants_real && r.n <= k) || (wants_recall && s.n <= k) ||
      !finite(r) || !finite(s))
    return {nan, nan, nan, nan};
  auto rr = wants_real ? radii(r, k, threads) : std::vector<T>{};
  auto sr = wants_recall ? radii(s, k, threads) : std::vector<T>{};
  std::vector<int64_t> counts(wants_counts ? s.n : 0);
  auto recall = std::make_unique<std::atomic<uint8_t>[]>(wants_recall ? r.n : 0),
       cover = std::make_unique<std::atomic<uint8_t>[]>(wants_cover ? r.n : 0);
  for (int64_t i = 0; i < r.n; ++i) {
    if (wants_recall)
      recall[i] = 0;
    if (wants_cover)
      cover[i] = 0;
  }
#pragma omp parallel for num_threads(threads) if (threads > 1)
  for (int64_t j = 0; j < s.n; ++j) {
    alignas(64) T distances[128];
    int64_t count = 0;
    for (int64_t i = 0; i < r.n; i += 128) {
      int64_t length = std::min<int64_t>(128, r.n - i);
      block(s, j, r, i, length, distances);
      for (int64_t z = 0; z < length; ++z) {
        T d = distances[z];
        if (wants_real && d < rr[i + z]) {
          if (wants_counts)
            ++count;
          if (wants_cover)
            cover[i + z].store(1, std::memory_order_relaxed);
        }
        if (wants_recall && d < sr[j])
          recall[i + z].store(1, std::memory_order_relaxed);
      }
    }
    if (wants_counts)
      counts[j] = count;
  }
  int64_t precision = 0, density = 0, recalled = 0, covered = 0;
  for (auto n : counts) {
    precision += n > 0;
    density += n;
  }
  for (int64_t i = 0; i < r.n; ++i) {
    if (wants_recall)
      recalled += recall[i].load();
    if (wants_cover)
      covered += cover[i].load();
  }
  return {(components & 1) ? double(precision) / s.n : nan,
          wants_recall ? double(recalled) / r.n : nan,
          (components & 4) ? double(density) / (double(k) * s.n) : nan,
          wants_cover ? double(covered) / r.n : nan};
}
template std::vector<double> prdc(const Sample<float>&, const Sample<float>&, int64_t, int,
                                  int64_t);
template std::vector<double> prdc(const Sample<double>&, const Sample<double>&, int64_t, int,
                                  int64_t);
} // namespace ltm
