#pragma once
#include "cpu/runtime/types.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace ltm {
/** @brief Draw-owned column-major numeric/category buffers.
 * @tparam T Selected float or double arithmetic type.
 * Numeric cells preserve nonfinite observations; categorical -1 denotes
 * missing. Repeated source indices remain repeated rows. No storage aliases
 * input tensors.
 */
template <class T> struct Sample {
  std::vector<std::vector<T>> numeric;
  std::vector<std::vector<int64_t>> categorical;
  std::vector<uint8_t> complete;
  int64_t n = 0;
  int64_t width() const {
    return numeric.size() + categorical.size();
  }
};
/** @brief Copy validated source positions into owned column-major draw buffers,
 * preserving row order, missingness and duplicate multiplicity. */
template <class T>
Sample<T> gather(const Tensor& x, const Tensor& c, const Tensor& mask, const int64_t* indices,
                 int64_t n) {
  Sample<T> s;
  s.n = n;
  s.numeric.resize(x.size(1));
  s.categorical.resize(c.size(1));
  s.complete.resize(n);
  const auto* xp = x.const_data_ptr<T>();
  const auto* cp = c.const_data_ptr<int64_t>();
  const auto* mp = mask.const_data_ptr<bool>();
  for (int64_t j = 0; j < x.size(1); ++j) {
    auto& v = s.numeric[j];
    v.resize(n);
    for (int64_t i = 0; i < n; ++i)
      v[i] = xp[indices[i] * x.size(1) + j];
  }
  for (int64_t j = 0; j < c.size(1); ++j) {
    auto& v = s.categorical[j];
    v.resize(n);
    for (int64_t i = 0; i < n; ++i)
      v[i] = cp[indices[i] * c.size(1) + j];
  }
  for (int64_t i = 0; i < n; ++i)
    s.complete[i] = mp[indices[i]];
  return s;
}
template <class T>
Sample<T> select(const Sample<T>& s, const std::vector<int64_t>& rows, int64_t exclude = -1) {
  Sample<T> out;
  out.n = rows.size();
  for (int64_t j = 0; j < (int64_t)s.numeric.size(); ++j)
    if (j != exclude) {
      std::vector<T> v;
      v.reserve(rows.size());
      for (auto i : rows)
        v.push_back(s.numeric[j][i]);
      out.numeric.push_back(std::move(v));
    }
  for (int64_t j = 0; j < (int64_t)s.categorical.size(); ++j)
    if (j + (int64_t)s.numeric.size() != exclude) {
      std::vector<int64_t> v;
      v.reserve(rows.size());
      for (auto i : rows)
        v.push_back(s.categorical[j][i]);
      out.categorical.push_back(std::move(v));
    }
  out.complete.assign(out.n, 1);
  return out;
}
template <class T> Sample<T> raw(const Tensor& x, const Tensor& c) {
  Sample<T> s;
  s.n = x.size(0);
  s.numeric.resize(x.size(1));
  s.categorical.resize(c.size(1));
  s.complete.assign(s.n, 1);
  const T* xp = x.const_data_ptr<T>();
  const int64_t* cp = c.const_data_ptr<int64_t>();
  for (int64_t j = 0; j < x.size(1); ++j) {
    s.numeric[j].resize(s.n);
    for (int64_t i = 0; i < s.n; ++i)
      s.numeric[j][i] = xp[i * x.size(1) + j];
  }
  for (int64_t j = 0; j < c.size(1); ++j) {
    s.categorical[j].resize(s.n);
    for (int64_t i = 0; i < s.n; ++i)
      s.categorical[j][i] = cp[i * c.size(1) + j];
  }
  return s;
}

} // namespace ltm
