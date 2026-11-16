#pragma once

#include "cuda/linalg/factorization.hpp"
#include "cuda/linalg/linear.hpp"
#include <cmath>
#include <vector>

namespace ltm::cuda::classifier::preparation {
using at::Tensor;
// Construct P = D + L L^T from the shifted Nyström sketch, using the same
// factorization for preparation and independent numerical/failure oracles.
// Returns inverse root, QR reflectors/tau, small rotations/ratios, cached basis,
// residual diagonal, and a CPU int64 [1] fallback indicator.
inline std::vector<Tensor> nystrom_factor(const Tensor& diagonal, const Tensor& omega,
                                          const Tensor& images, double shift) {
  linalg::matrix(omega);
  linalg::matrix(images);
  TORCH_CHECK(omega.sizes() == images.sizes() && omega.device() == images.device() &&
                  omega.scalar_type() == images.scalar_type() && diagonal.is_cuda() &&
                  diagonal.device() == omega.device() &&
                  diagonal.scalar_type() == omega.scalar_type() && diagonal.dim() == 1 &&
                  diagonal.size(0) == omega.size(0) && omega.size(0) >= omega.size(1) &&
                  std::isfinite(shift) && shift > 0,
              "invalid native Nyström factor inputs");
  c10::cuda::CUDAGuard guard(omega.device());
  TORCH_CHECK_LINALG(at::isfinite(diagonal).all().item<bool>() && diagonal.gt(0).all().item<bool>(),
                     "SketchySAGA requires a positive finite curvature diagonal");
  const int64_t d = omega.size(0), rank = omega.size(1);
  const bool single = omega.scalar_type() == at::kFloat;
  auto opts = omega.options();
  Tensor inverse_root, reflectors, tau, rotations, ratios, basis, residual;
  int64_t fallback = 0;
  try {
    auto shifted = images + shift * omega;
    auto core = linalg::gemm(omega.transpose(0, 1), shifted);
    auto lower = linalg::lower_factor((core + core.transpose(0, 1)) * 0.5);
    auto skinny = linalg::triangular(lower, shifted.transpose(0, 1)).transpose(0, 1);
    auto sv = linalg::svd(skinny);
    auto lowrank =
        std::get<0>(sv) * (std::get<1>(sv).square() - shift).clamp_min(0).sqrt().unsqueeze(0);
    residual = at::maximum(diagonal - lowrank.square().sum(1), diagonal * (single ? 1e-6 : 1e-10));
    inverse_root = residual.rsqrt();
    auto decomposition = linalg::qr(inverse_root.unsqueeze(1) * lowrank);
    reflectors = std::get<1>(decomposition);
    tau = std::get<2>(decomposition);
    auto small = linalg::svd(at::triu(reflectors.narrow(0, 0, rank)));
    rotations = std::get<0>(small);
    ratios = (1 + std::get<1>(small).square()).reciprocal();
    TORCH_CHECK_LINALG(at::isfinite(ratios).all().item<bool>() &&
                           at::isfinite(inverse_root).all().item<bool>(),
                       "Nonfinite Nyström factor");
    basis = linalg::gemm(std::get<0>(decomposition), rotations);
  } catch (const c10::LinAlgError&) {
    fallback = 1;
    residual = diagonal;
    inverse_root = diagonal.rsqrt();
    reflectors = at::zeros({1, d}, opts).transpose(0, 1);
    tau = at::zeros({1}, opts);
    rotations = at::ones({1, 1}, opts);
    ratios = at::ones({1}, opts);
    basis = at::zeros({1, d}, opts).transpose(0, 1);
  }
  auto fallback_flag =
      at::tensor({fallback}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  return {inverse_root, reflectors, tau, rotations, ratios, basis, residual, fallback_flag};
}
} // namespace ltm::cuda::classifier::preparation
