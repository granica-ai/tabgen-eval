#include "cpu/classifier/detail/saga.h"
#include "cpu/runtime/tensor_validation.h"

namespace ltm::saga::detail {
void valid(const Design& x) {
  numbers(x.numbers);
  matrix(x.identities, at::kLong);
  TORCH_CHECK(x.n() > 0 && x.identities.size(0) == x.n() && x.d() > 1,
              "invalid compact SAGA design dimensions");
  for (const auto& t : {x.centers, x.scales})
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == x.numbers.scalar_type() && t.dim() == 1 &&
                    t.is_contiguous() && t.numel() == x.k() && at::isfinite(t).all().item<bool>(),
                "invalid categorical transform");
  TORCH_CHECK(at::isfinite(x.numbers).all().item<bool>() && (x.scales > 0).all().item<bool>(),
              "nonfinite SAGA design or scale");
  const auto* ids = x.identities.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < x.identities.numel(); ++i)
    TORCH_CHECK(ids[i] >= -1 && ids[i] < x.k(), "invalid category feature identity");
}
Tensor adjoint(const Design& x, const Tensor& v) {
  return AT_DISPATCH_FLOATING_TYPES(x.numbers.scalar_type(), "cpu_classifier", [&]() {
    auto out = at::zeros({x.d()}, x.numbers.options());
    if (x.pn())
      out.narrow(0, 0, x.pn()).copy_(at::mv(x.numbers.t(), v));
    auto* p = out.data_ptr<scalar_t>();
    const auto* values = v.const_data_ptr<scalar_t>();
    const auto* ids = x.identities.const_data_ptr<int64_t>();
    double total = v.sum().item<double>();
    // Feature IDs of different source columns occupy disjoint ranges. Serial
    // accumulation retains row order, needs only compact minibatch identities.
    for (int64_t i = 0; i < x.n(); ++i)
      for (int64_t c = 0; c < x.pc(); ++c) {
        auto id = ids[i * x.pc() + c];
        if (id >= 0)
          p[x.pn() + id] += values[i];
      }
    const auto* center = x.centers.const_data_ptr<scalar_t>();
    const auto* scale = x.scales.const_data_ptr<scalar_t>();
    for (int64_t j = 0; j < x.k(); ++j)
      p[x.pn() + j] = (p[x.pn() + j] - center[j] * total) / scale[j];
    p[x.d() - 1] = total;
    return out;
  });
}
Tensor hessian(const Design& x, const Tensor& v, double ridge) {
  auto out = adjoint(x, mv(x, v) / (4 * x.n()));
  out.narrow(0, 0, x.d() - 1).add_(v.narrow(0, 0, x.d() - 1), ridge);
  return out;
}
Tensor diagonal(const Design& x, double ridge) {
  return AT_DISPATCH_FLOATING_TYPES(x.numbers.scalar_type(), "cpu_classifier", [&]() {
    auto out = at::zeros({x.d()}, x.numbers.options());
    auto* p = out.data_ptr<scalar_t>();
    const auto* numeric = x.numbers.const_data_ptr<scalar_t>();
    for (int64_t i = 0; i < x.n(); ++i)
      for (int64_t j = 0; j < x.pn(); ++j) {
        double value = numeric[i * x.pn() + j];
        p[j] += value * value;
      }
    for (int64_t j = 0; j < x.pn(); ++j)
      p[j] /= 4 * x.n();
    const auto* ids = x.identities.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < x.identities.numel(); ++i)
      if (ids[i] >= 0)
        p[x.pn() + ids[i]] += 1.;
    const auto* centers = x.centers.const_data_ptr<scalar_t>();
    const auto* scales = x.scales.const_data_ptr<scalar_t>();
    for (int64_t j = 0; j < x.k(); ++j)
      p[x.pn() + j] = (p[x.pn() + j] * (1 - 2 * centers[j]) + x.n() * centers[j] * centers[j]) /
                      (4 * x.n() * scales[j] * scales[j]);
    p[x.d() - 1] = .25;
    out.narrow(0, 0, x.d() - 1).add_(ridge);
    return out;
  });
}

} // namespace ltm::saga::detail

namespace ltm::saga {
Tensor mv(const Design& x, const Tensor& weights) {
  return AT_DISPATCH_FLOATING_TYPES(x.numbers.scalar_type(), "cpu_classifier", [&]() {
    auto out = x.pn() ? at::mv(x.numbers, weights.narrow(0, 0, x.pn()))
                      : at::zeros({x.n()}, x.numbers.options());
    auto cats = weights.narrow(0, x.pn(), x.k()) / x.scales;
    double offset =
        weights.select(0, x.d() - 1).item<double>() - at::dot(x.centers, cats).item<double>();
    auto* values = out.data_ptr<scalar_t>();
    const auto* ids = x.identities.const_data_ptr<int64_t>();
    const auto* c = cats.const_data_ptr<scalar_t>();
    for (int64_t i = 0; i < x.n(); ++i) {
      double value = offset;
      for (int64_t j = 0; j < x.pc(); ++j) {
        auto id = ids[i * x.pc() + j];
        if (id >= 0)
          value += c[id];
      }
      values[i] += value;
    }
    return out;
  });
}

} // namespace ltm::saga
