#include "cpu/classifier/detail/newton.h"
#include "cpu/runtime/thread_budget.h"
#include <ATen/Config.h>
#if defined(__linux__)
#include <ATen/native/BatchLinearAlgebra.h>
#endif
#include <ATen/record_function.h>
#include <optional>

namespace ltm::classifier_detail {
namespace {
double scalar_dot(const Tensor& a, const Tensor& b) {
  return at::dot(a, b).item<double>();
}

} // namespace

double objective(const Tensor& x, const Tensor& y, const Tensor& w, double ridge) {
  RECORD_FUNCTION("tabgen_eval_cpu::c2st_objective", std::vector<c10::IValue>());
  auto z = at::mv(x, w);
  auto signed_z = at::where(y == 1, -z, z);
  auto loss = at::clamp_min(signed_z, 0) + at::log1p(at::exp(-at::abs(signed_z)));
  auto penalized = w.narrow(0, 0, w.numel() - 1);
  return loss.mean(at::kDouble).item<double>() + 0.5 * ridge * scalar_dot(penalized, penalized);
}

std::pair<Tensor, Tensor> derivatives(const Tensor& x, const Tensor& y, const Tensor& w,
                                      double ridge) {
  RECORD_FUNCTION("tabgen_eval_cpu::c2st_derivatives", std::vector<c10::IValue>());
  auto z = at::mv(x, w);
  auto e = at::exp(-at::abs(z));
  auto denominator = 1 + e;
  auto probability = at::where(z >= 0, 1 / denominator, e / denominator);
  auto residual =
      at::where(y == 0, probability, at::where(z >= 0, -e / denominator, -1 / denominator));
  auto gradient = at::mv(x.transpose(0, 1), residual / x.size(0));
  gradient.narrow(0, 0, w.numel() - 1).add_(w.narrow(0, 0, w.numel() - 1), ridge);
  return {gradient, e / (denominator * denominator * x.size(0))};
}

namespace {
Tensor gram(const Tensor& design, double ridge) {
  RECORD_FUNCTION("tabgen_eval_cpu::c2st_gram", std::vector<c10::IValue>());
  int64_t d = design.size(1);
  Tensor out;
  // Compact parallel sketches benefit from one large GEMM: small tiled
  // products repeatedly enter BLAS and copy their output. Tall direct designs
  // and serial fits retain the lower-triangle work saving.
  if (d < 256 || (omp_get_max_threads() > 1 && d <= 1024 && design.size(0) <= 4 * d)) {
    out = at::mm(design.transpose(0, 1), design);
  } else {
    // Form one triangle through ATen GEMM blocks and mirror it. A full d-by-d
    // GEMM performs almost twice the needed Hessian work on wide designs.
    out = at::empty({d, d}, design.options());
    constexpr int64_t tile = 128;
    for (int64_t i = 0; i < d; i += tile) {
      int64_t ni = std::min(tile, d - i);
      auto a = design.narrow(1, i, ni);
      for (int64_t j = 0; j <= i; j += tile) {
        int64_t nj = std::min(tile, d - j);
        auto value = at::mm(a.transpose(0, 1), design.narrow(1, j, nj));
        out.narrow(0, i, ni).narrow(1, j, nj).copy_(value);
        if (i != j)
          out.narrow(0, j, nj).narrow(1, i, ni).copy_(value.transpose(0, 1));
      }
    }
  }
  out.diagonal().narrow(0, 0, out.size(0) - 1).add_(ridge);
  return out;
}

struct Factor {
  Tensor lower;
  bool damped;
};

std::optional<Factor> factor(const Tensor& matrix) {
  RECORD_FUNCTION("tabgen_eval_cpu::c2st_factor", std::vector<c10::IValue>());
  double scale = std::max(1.0, matrix.diagonal().abs().max().item<double>());
  const std::vector<double> damping = matrix.scalar_type() == at::kFloat
                                          ? std::vector<double>{0., 1e-7, 1e-6, 1e-5, 1e-4}
                                          : std::vector<double>{0., 1e-12, 1e-10, 1e-8, 1e-6};
  for (double multiplier : damping) {
    auto trial = matrix.clone();
    if (multiplier)
      trial.diagonal().add_(multiplier * scale);
    auto result = at::linalg_cholesky_ex(trial, false, false);
    if (std::get<1>(result).item<int64_t>() == 0) {
      auto lower = std::get<0>(result);
      return Factor{lower, multiplier > 0};
    }
  }
  return std::nullopt;
}

Tensor solve(const Factor& factor, const Tensor& rhs) {
  RECORD_FUNCTION("tabgen_eval_cpu::c2st_solve", std::vector<c10::IValue>());
  // Linux Torch builds export this BLAS wrapper. Apple Silicon builds may not;
  // use the public triangular-solve operators below on those builds.
  // Reuse one validated FP32/FP64 factor for both substitutions.
#if defined(__linux__) && AT_BUILD_WITH_BLAS()
  int64_t d = rhs.numel();
  if (factor.lower.stride(0) == 1 && factor.lower.stride(1) <= INT32_MAX && d <= INT32_MAX) {
    // A small single-RHS solve has limited parallel work. Keep its BLAS
    // budget local; the surrounding Gram products retain the fit budget.
    std::optional<ThreadBudget> serial;
    if (d <= 1024)
      serial.emplace(1);
    auto out = rhs.clone();
    // BLAS accepts a mutable pointer but does not modify the triangular matrix.
    AT_DISPATCH_FLOATING_TYPES(rhs.scalar_type(), "cpu_newton_solve", [&]() {
      auto* lower = const_cast<scalar_t*>(factor.lower.const_data_ptr<scalar_t>());
      auto* values = out.data_ptr<scalar_t>();
      at::native::blasTriangularSolve<scalar_t>('L', 'L', 'N', 'N', d, 1, lower,
                                                factor.lower.stride(1), values, d);
      at::native::blasTriangularSolve<scalar_t>('L', 'L', 'T', 'N', d, 1, lower,
                                                factor.lower.stride(1), values, d);
    });
    return out;
  }
#endif
  auto intermediate = at::linalg_solve_triangular(factor.lower, rhs.unsqueeze(1), false);
  return at::linalg_solve_triangular(factor.lower.transpose(0, 1), intermediate, true).squeeze(1);
}

} // namespace

Fit fit(const Tensor& x, const Tensor& y, const Options& o) {
  int64_t d = x.size(1);
  Fit output;
  output.coefficients = at::zeros({d}, x.options());
  output.solver = "direct";
  for (int64_t iteration = 0; iteration < o.max_iter; ++iteration) {
    auto [gradient, curvature] = derivatives(x, y, output.coefficients, o.ridge);
    output.gradient_squared = scalar_dot(gradient, gradient);
    if (!std::isfinite(output.gradient_squared)) {
      output.reason = "nonfinite_gradient";
      break;
    }
    if (output.gradient_squared <= o.tolerance * o.tolerance) {
      output.converged = true;
      output.reason = "gradient_squared";
      break;
    }
    auto design = x * curvature.sqrt().unsqueeze(1);
    auto decomposition = factor(gram(design, o.ridge));
    if (!decomposition) {
      output.reason = "factorization_failed";
      break;
    }
    auto step = solve(*decomposition, gradient);
    double descent = scalar_dot(gradient, step);
    // Diagnostic only: convergence requires the full squared gradient norm.
    output.decrement_squared = descent;
    if (!std::isfinite(descent) || descent <= 0) {
      output.reason = "invalid_direction";
      break;
    }
    double loss = objective(x, y, output.coefficients, o.ridge), rate = 1.0;
    bool accepted = false;
    for (int i = 0; i < 40; ++i) {
      auto candidate = output.coefficients - rate * step;
      double next = objective(x, y, candidate, o.ridge);
      if (std::isfinite(next) && next <= loss - 1e-4 * rate * descent) {
        output.coefficients = candidate;
        ++output.iterations;
        accepted = true;
        break;
      }
      rate *= 0.5;
    }
    if (!accepted) {
      output.reason = "line_search_failed";
      break;
    }
  }
  auto gradient = derivatives(x, y, output.coefficients, o.ridge).first;
  output.gradient_squared = scalar_dot(gradient, gradient);
  if (std::isfinite(output.gradient_squared) &&
      output.gradient_squared <= o.tolerance * o.tolerance) {
    output.converged = true;
    output.reason = "gradient_squared";
  }
  return output;
}

} // namespace ltm::classifier_detail
