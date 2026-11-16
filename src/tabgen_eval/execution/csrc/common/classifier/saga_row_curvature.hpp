#pragma once
#include <ATen/ATen.h>
#include <limits>

namespace ltm::saga::calibration {
// Compact row quadratic forms, on CPU or CUDA, with O(rows * rank) scratch.
// P^-1 = diag(root) [I + B diag(ratios-1) B'] diag(root).
// No one-hot matrix or per-row Householder applications are materialized.
inline at::Tensor
row_curvature(const at::Tensor& numbers, const at::Tensor& identities, const at::Tensor& centers,
              const at::Tensor& scales, const at::Tensor& root, const at::Tensor& basis,
              const at::Tensor& ratios,
              at::Tensor (*product)(const at::Tensor&, const at::Tensor&) = nullptr) {
  const int64_t m = numbers.size(0), pn = numbers.size(1), k = centers.numel();
  auto root2 = root.square();
  auto diagonal =
      pn ? at::mv(numbers.square(), root2.narrow(0, 0, pn)) : at::zeros({m}, numbers.options());
  auto weighted = root.unsqueeze(1) * basis;
  auto projection = pn ? (product ? product(numbers, weighted.narrow(0, 0, pn).contiguous())
                                  : at::mm(numbers, weighted.narrow(0, 0, pn)))
                       : at::zeros({m, ratios.numel()}, numbers.options());
  if (k) {
    auto cat_root2 = root2.narrow(0, pn, k);
    diagonal.add_((centers.square() / scales.square() * cat_root2).sum());
    auto correction = (1 - 2 * centers) / scales.square() * cat_root2;
    auto cat_basis = weighted.narrow(0, pn, k) / scales.unsqueeze(1);
    projection.sub_(at::mv(cat_basis.t(), centers));
    for (int64_t j = 0; j < identities.size(1); ++j) {
      auto ids = identities.select(1, j);
      auto valid = ids.ge(0), safe = ids.clamp_min(0);
      diagonal.add_(at::where(valid, correction.index_select(0, safe), 0));
      projection.add_(at::where(valid.unsqueeze(1), cat_basis.index_select(0, safe), 0));
    }
  }
  diagonal.add_(root2.select(0, root.numel() - 1));
  projection.add_(weighted.select(0, root.numel() - 1));
  auto quadratic = diagonal + at::mv(projection.square(), ratios - 1);
  const double epsilon = numbers.scalar_type() == at::kDouble
                             ? std::numeric_limits<double>::epsilon()
                             : std::numeric_limits<float>::epsilon();
  TORCH_CHECK_LINALG(at::isfinite(quadratic).all().item<bool>() &&
                         quadratic.ge(-diagonal * (64 * epsilon)).all().item<bool>(),
                     "nonfinite or negative preconditioned row curvature");
  // A tiny negative from subtractive roundoff cannot increase the sampled max.
  return quadratic.clamp_min(0) * .25;
}
inline double ridge_bound(const at::Tensor& root, double ridge) {
  return ridge * root.narrow(0, 0, root.numel() - 1).square().max().item<double>();
}
} // namespace ltm::saga::calibration
