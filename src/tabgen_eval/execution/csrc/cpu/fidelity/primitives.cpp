#include "cpu/fidelity/primitives.h"
#include "cpu/fidelity/fidelity.h"
#include "cpu/runtime/tensor_validation.h"
#include "cpu/runtime/thread_budget.h"

namespace ltm {
/** @brief Return a.T*b as [P,Q] in selected precision for contiguous CPU
 * [N,P]/[N,Q] inputs; read-only inputs and locally restored numerical thread
 * budget. */
Tensor gram(const Tensor& a, const Tensor& b) {
  numbers(a);
  numbers(b);
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.size(0) == b.size(0),
              "incompatible Gram layouts");
  ThreadBudget budget(1);
  return at::mm(a.transpose(0, 1), b);
}
/** @brief Return exact CPU int64 [P,P] overlap counts from contiguous bool
 * [N,P] masks without mutating inputs. */
Tensor pair_counts(const Tensor& x) {
  matrix(x, at::kBool);
  int64_t n = x.size(0), p = x.size(1);
  auto out = at::zeros({p, p}, x.options().dtype(at::kLong));
  auto* counts = out.data_ptr<int64_t>();
  const auto* mask = x.const_data_ptr<bool>();
  for (int64_t i = 0; i < p; ++i)
    for (int64_t j = i; j < p; ++j) {
      int64_t count = 0;
#pragma omp simd reduction(+ : count)
      for (int64_t r = 0; r < n; ++r)
        count += mask[r * p + i] && mask[r * p + j];
      counts[i * p + j] = counts[j * p + i] = count;
    }
  return out;
}
/** @brief Return int64 [pairs,N] occupied joint keys from contiguous int64
 * codes/pair indices; excluded codes map to -1 and overflow raises before
 * encoding. */
Tensor keys(const Tensor& c, const Tensor& pairs, const Tensor& widths) {
  matrix(c, at::kLong);
  matrix(pairs, at::kLong);
  TORCH_CHECK(pairs.size(1) == 2 && widths.device().is_cpu() && widths.scalar_type() == at::kLong &&
                  widths.dim() == 1 && widths.numel() == pairs.size(0) && widths.is_contiguous(),
              "invalid joint layout");
  int64_t n = c.size(0), p = c.size(1), b = pairs.size(0);
  auto out = at::empty({b, n}, c.options());
  auto* dst = out.data_ptr<int64_t>();
  const auto* src = c.const_data_ptr<int64_t>();
  const auto* idx = pairs.const_data_ptr<int64_t>();
  const auto* w = widths.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < b; ++i) {
    TORCH_CHECK(idx[2 * i] >= 0 && idx[2 * i] < p && idx[2 * i + 1] >= 0 && idx[2 * i + 1] < p &&
                    w[i] > 0,
                "invalid joint pair/width");
    for (int64_t r = 0; r < n; ++r) {
      auto a = src[r * p + idx[2 * i]], v = src[r * p + idx[2 * i + 1]];
      TORCH_CHECK(a < 0 || a <= (INT64_MAX - std::max<int64_t>(v, 0)) / w[i], "joint key overflow");
      dst[i * n + r] = a >= 0 && v >= 0 ? a * w[i] + v : -1;
    }
  }
  return out;
}
/** @brief Return exact int64 [pairs,grid] histograms for bounded grids; valid
 * keys must fit grid and input buffers remain read-only. */
Tensor dense(const Tensor& c, const Tensor& pairs, const Tensor& widths, int64_t grid) {
  TORCH_CHECK(grid >= 1 && grid <= 4096, "invalid dense grid");
  auto k = keys(c, pairs, widths);
  auto out = at::zeros({pairs.size(0), grid}, c.options());
  const auto* kp = k.const_data_ptr<int64_t>();
  auto* p = out.data_ptr<int64_t>();
  for (int64_t i = 0; i < pairs.size(0); ++i)
    for (int64_t j = 0; j < c.size(0); ++j)
      if (kp[i * c.size(0) + j] >= 0) {
        TORCH_CHECK(kp[i * c.size(0) + j] < grid, "joint key outside dense grid");
        ++p[i * grid + kp[i * c.size(0) + j]];
      }
  return out;
}
template <class T>
Tensor sorted_tv_t(const Tensor& a, const Tensor& b, const Tensor& na, const Tensor& nb,
                   const Tensor& prototype) {
  auto out = at::empty({a.size(0)}, prototype.options());
  const auto* ap = a.const_data_ptr<int64_t>();
  const auto* bp = b.const_data_ptr<int64_t>();
  const auto* ar = na.const_data_ptr<int64_t>();
  const auto* bs = nb.const_data_ptr<int64_t>();
  auto* dst = out.data_ptr<T>();
  for (int64_t p = 0; p < a.size(0); ++p) {
    if (!ar[p] || !bs[p]) {
      dst[p] = T(nan);
      continue;
    }
    int64_t i = 0, j = 0;
    T sum = 0, correction = 0;
    while (i < a.size(1) && ap[p * a.size(1) + i] < 0)
      ++i;
    while (j < b.size(1) && bp[p * b.size(1) + j] < 0)
      ++j;
    while (i < a.size(1) || j < b.size(1)) {
      int64_t key =
          j == b.size(1) || (i < a.size(1) && ap[p * a.size(1) + i] < bp[p * b.size(1) + j])
              ? ap[p * a.size(1) + i]
              : bp[p * b.size(1) + j];
      int64_t starti = i, startj = j;
      while (i < a.size(1) && ap[p * a.size(1) + i] == key)
        ++i;
      while (j < b.size(1) && bp[p * b.size(1) + j] == key)
        ++j;
      T delta = std::abs(T(i - starti) / T(ar[p]) - T(j - startj) / T(bs[p]));
      T y = delta - correction, v = sum + y;
      correction = (v - sum) - y;
      sum = v;
    }
    dst[p] = sum;
  }
  return out;
}
/** @brief Return selected-prototype-precision [pairs] TV complements from
 * sorted int64 keys and valid counts; empty marginals yield NaN. */
Tensor sorted_tv(const Tensor& a, const Tensor& b, const Tensor& na, const Tensor& nb,
                 const Tensor& proto) {
  matrix(a, at::kLong);
  matrix(b, at::kLong);
  TORCH_CHECK(a.size(0) == b.size(0) && na.device().is_cpu() && nb.device().is_cpu() &&
                  na.scalar_type() == at::kLong && nb.scalar_type() == at::kLong &&
                  na.numel() == a.size(0) && nb.numel() == a.size(0) && na.is_contiguous() &&
                  nb.is_contiguous() && proto.device().is_cpu() && proto.dim() == 0,
              "invalid TV inputs");
  TORCH_CHECK(proto.scalar_type() == at::kFloat || proto.scalar_type() == at::kDouble,
              "invalid TV dtype");
  return proto.scalar_type() == at::kDouble ? sorted_tv_t<double>(a, b, na, nb, proto)
                                            : sorted_tv_t<float>(a, b, na, nb, proto);
}
/** @brief Validate full-source Conditional rare applicability in selected
 * precision; malformed numeric/category metadata raises, and empty retained
 * rare sets return false. */
bool rare(const Tensor& x, const Tensor& c, int64_t minimum, int64_t bins, double threshold) {
  numbers(x);
  matrix(c, at::kLong);
  TORCH_CHECK(x.size(0) == c.size(0) && minimum > 0 && bins > 0 && std::isfinite(threshold) &&
                  threshold >= 0 && threshold <= 1,
              "invalid rare-slice parameters");
  return x.scalar_type() == at::kDouble
             ? has_rare_sample(raw<double>(x, c), minimum, bins, threshold)
             : has_rare_sample(raw<float>(x, c), minimum, bins, threshold);
}

} // namespace ltm
