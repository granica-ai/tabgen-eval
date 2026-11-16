#include "cpu/classifier/detail/saga.h"

namespace ltm::saga::detail {
Tensor gaussian(int64_t n, int64_t k, const at::TensorOptions& opts, std::mt19937_64& rng) {
  auto out = at::empty({n, k}, opts);
  std::normal_distribution<double> normal;
  AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "cpu_sketch_gaussian", [&]() {
    for (int64_t i = 0; i < out.numel(); ++i)
      out.data_ptr<scalar_t>()[i] = normal(rng);
  });
  return out;
}
Inverse preconditioner(const Design& x, double ridge, int64_t rank, std::mt19937_64& rng) {
  auto diag = diagonal(x, ridge);
  auto omega =
      std::get<0>(at::linalg_qr(gaussian(x.d(), rank, x.numbers.options(), rng), "reduced"));
  auto y = at::empty({x.d(), rank}, x.numbers.options());
  for (int64_t i = 0; i < rank; ++i)
    y.select(1, i).copy_(hessian(x, omega.select(1, i).contiguous(), ridge));
  const bool fp32 = x.numbers.scalar_type() == at::kFloat;
  const double epsilon =
      fp32 ? std::numeric_limits<float>::epsilon() : std::numeric_limits<double>::epsilon();
  double shift = std::max(std::numeric_limits<double>::min(), at::norm(y).item<double>() * epsilon);
  auto shifted = y + shift * omega;
  auto core = at::mm(omega.t(), shifted);
  core = (core + core.t()) * .5;
  auto lower = at::linalg_cholesky(core);
  auto skinny = at::linalg_solve_triangular(lower, shifted.t(), false).t();
  auto svd = at::linalg_svd(skinny, false);
  auto low =
      std::get<0>(svd) * (std::get<1>(svd).square() - shift).clamp_min(0).sqrt().unsqueeze(0);
  auto residual = at::maximum(diag - low.square().sum(1), diag * (fp32 ? 1e-6 : 1e-10));
  Inverse p;
  p.root = residual.rsqrt();
  auto qr = at::geqrf(p.root.unsqueeze(1) * low);
  p.reflectors = std::get<0>(qr);
  p.tau = std::get<1>(qr);
  auto triangular = at::triu(p.reflectors.narrow(0, 0, rank));
  auto small = at::linalg_svd(triangular, false);
  p.rotations = std::get<0>(small);
  p.ratios = 1 / (1 + std::get<1>(small).square());
  auto q = at::orgqr(p.reflectors, p.tau);
  p.basis = at::mm(q, p.rotations);
  // Cached products avoid reflectors unless FP32 cancellation would be severe.
  p.stable = fp32 && p.ratios.min().item<double>() < 1e-3;
  return p;
}

} // namespace ltm::saga::detail
