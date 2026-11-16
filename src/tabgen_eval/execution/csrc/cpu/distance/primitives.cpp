#include "cpu/distance/primitives.h"
#include "cpu/runtime/tensor_validation.h"

namespace ltm {
template <class T>
Tensor distances_t(const Tensor& a, const Tensor& b, const Tensor& ca, const Tensor& cb) {
  auto out = at::zeros({a.size(0), b.size(0)}, a.options());
  auto* p = out.data_ptr<T>();
  auto* x = a.const_data_ptr<T>();
  auto* y = b.const_data_ptr<T>();
  auto* c = ca.const_data_ptr<int64_t>();
  auto* d = cb.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < a.size(0); ++i)
    for (int64_t j = 0; j < b.size(0); ++j) {
      T value = 0;
      int64_t cat = 0;
      for (int64_t k = 0; k < a.size(1); ++k) {
        T diff = x[i * a.size(1) + k] - y[j * b.size(1) + k];
        value += diff * diff;
      }
      for (int64_t k = 0; k < ca.size(1); ++k) {
        auto u = c[i * ca.size(1) + k], v = d[j * cb.size(1) + k];
        cat += 2 * (u != v) - ((u < 0) != (v < 0));
      }
      p[i * b.size(0) + j] = value + T(cat);
    }
  return out;
}
/** @brief Return [Q,R] direct squared distances in input float/double dtype.
 * Inputs are contiguous CPU [Q/R,PN] numeric and int64 [Q/R,PC] codes; inputs
 * are read-only and negative identities preserve unknown-zero-one-hot
 * semantics. */
Tensor distances(const Tensor& a, const Tensor& b, const Tensor& ca, const Tensor& cb) {
  numbers(a);
  numbers(b);
  matrix(ca, at::kLong);
  matrix(cb, at::kLong);
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.size(1) == b.size(1) &&
                  ca.size(1) == cb.size(1) && a.size(0) == ca.size(0) && b.size(0) == cb.size(0),
              "incompatible distance layouts");
  return a.scalar_type() == at::kDouble ? distances_t<double>(a, b, ca, cb)
                                        : distances_t<float>(a, b, ca, cb);
}

} // namespace ltm
