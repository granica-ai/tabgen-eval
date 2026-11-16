/** @file
 * @brief Dense CUDA Newton iteration, damping and stable Armijo control.
 * ATen supplies storage only. Numerical work uses dedicated device kernels and
 * direct cuBLAS/cuSOLVER calls on the caller's stream in declared FP32/FP64.
 */
#include "cuda/classifier/newton.hpp"
#include "cuda/linalg/dense_solver.hpp"
#include "cuda/runtime/blas_scope.hpp"
#include "cuda/runtime/ptx.hpp"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace ltm::cuda::classifier::newton {
using at::Tensor;
namespace {
int ceildiv(int64_t n) {
  return int((n + 255) / 256);
}
void check(cudaError_t value) {
  runtime::check_runtime(value, "CUDA Newton error: ");
}
std::shared_ptr<runtime::KernelGroup> kernels(const std::string& path) {
  return runtime::kernels(path, "newton",
                          {"newton_logistic", "newton_gradient", "newton_reduce",
                           "newton_weight_design", "newton_ridge_scale", "newton_damping",
                           "newton_dot", "newton_candidate", "newton_loss_difference"},
                          "missing Newton PTX: ", "CUDA Newton driver error: ");
}

template <class Real>
Result fit_impl(const Tensor& design, const Tensor& labels, const std::string& ptx, double ridge,
                double tolerance, int64_t max_iter, int64_t limit) {
  c10::cuda::CUDAGuard device(design.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto module = kernels(ptx);
  const int n = design.size(0), d = design.size(1);
  const int rows = ceildiv(n), features = ceildiv(d), blocks = std::max(rows, features);
  auto blas = at::cuda::getCurrentCUDABlasHandle();
  auto solver = at::cuda::getCurrentCUDASolverDnHandle();
  runtime::BlasScope scope(blas, "CUDA Newton cuBLAS error: ");
  // Query before allocating model-sized scratch. POTRF's buffer query consumes
  // dimension/type metadata, not matrix values.
  auto probe = at::empty({1}, design.options());
  int count = linalg::dense_factor_workspace(solver, probe.data_ptr<Real>(), d);
  int64_t bytes =
      sizeof(Real) * (int64_t(n) * d + 2LL * d * d + 4LL * n + 5LL * d + std::max(1, count) + 1) +
      sizeof(double) * (2LL * blocks + 3) + sizeof(int);
  TORCH_CHECK_WITH(OutOfMemoryError, bytes <= limit,
                   "CUDA Newton buffers and cuSOLVER workspace exceed model allowance");
  auto opts = design.options();
  auto weights = at::empty({d}, opts), candidate = at::empty({d}, opts),
       change = at::empty({d}, opts), gradient = at::empty({d}, opts), step = at::empty({d}, opts);
  auto logits = at::empty({n}, opts), residual = at::empty({n}, opts), roots = at::empty({n}, opts),
       delta_logits = at::empty({n}, opts);
  auto weighted = at::empty({d, n}, opts).transpose(0, 1);
  auto matrix = at::empty({d, d}, opts), lower = at::empty({d, d}, opts);
  auto work = at::empty({std::max(1, count)}, opts);
  auto info = at::empty({1}, opts.dtype(at::kInt));
  auto doubles = opts.dtype(at::kDouble);
  auto parts = at::empty({blocks}, doubles), loss_parts = at::empty({blocks}, doubles),
       scalars = at::empty({3}, doubles);
  auto host = at::empty({3}, doubles.device(at::kCPU).pinned_memory(true));
  auto host_info = at::empty({1}, info.options().device(at::kCPU).pinned_memory(true));
  double* values = host.data_ptr<double>();
  check(cudaMemsetAsync(weights.data_ptr(), 0, d * sizeof(Real), stream));
  auto reduce = [&](const Tensor& partial, int length, int offset) {
    module->launch("newton_reduce", 1, 256, 0, stream, partial.const_data_ptr<double>(),
                   scalars.data_ptr<double>() + offset, length);
  };
  auto receive = [&](int length) {
    check(cudaMemcpyAsync(values, scalars.const_data_ptr(), length * sizeof(double),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
  };
  auto status_info = [&]() {
    check(cudaMemcpyAsync(host_info.data_ptr(), info.const_data_ptr(), sizeof(int),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    int status = *host_info.data_ptr<int>();
    TORCH_CHECK(status >= 0, "CUDA Newton cuSOLVER rejected argument ", -status);
    return status;
  };
  double squared = std::numeric_limits<double>::quiet_NaN();
  double decrement = squared, maximum_damping = 0, loss = squared;
  auto derivatives = [&]() {
    linalg::dense_mv(blas, design.const_data_ptr<Real>(), weights.const_data_ptr<Real>(),
                     logits.data_ptr<Real>(), n, d, false);
    module->launch("newton_logistic", rows, 256, 0, stream, logits.const_data_ptr<Real>(),
                   labels.const_data_ptr<Real>(), residual.data_ptr<Real>(), roots.data_ptr<Real>(),
                   loss_parts.data_ptr<double>(), n);
    linalg::dense_mv(blas, design.const_data_ptr<Real>(), residual.const_data_ptr<Real>(),
                     gradient.data_ptr<Real>(), n, d, true);
    module->launch("newton_gradient", features, 256, 0, stream, gradient.data_ptr<Real>(),
                   weights.const_data_ptr<Real>(), Real(ridge), parts.data_ptr<double>(), d);
    reduce(parts, features, 0);
    reduce(loss_parts, rows, 1);
    module->launch("newton_dot", ceildiv(d - 1), 256, 0, stream, weights.const_data_ptr<Real>(),
                   weights.const_data_ptr<Real>(), parts.data_ptr<double>(), d - 1);
    reduce(parts, ceildiv(d - 1), 2);
    receive(3);
    squared = values[0];
    loss = values[1] + 0.5 * ridge * values[2];
  };
  std::vector<double> damping = {0, 1e-12, 1e-10, 1e-8, 1e-6};
  if constexpr (std::is_same_v<Real, float>)
    for (int multiplier : {1, 10, 100, 1000})
      damping.push_back(32 * std::numeric_limits<Real>::epsilon() * multiplier);
  Status terminal = Status::max_iter;
  int64_t completed = 0;
  bool gradient_current = false;
  for (int64_t iteration = 0; iteration < max_iter; ++iteration) {
    derivatives();
    gradient_current = true;
    if (!std::isfinite(squared)) {
      terminal = Status::nonfinite_gradient;
      break;
    }
    if (squared <= tolerance * tolerance) {
      terminal = Status::gradient_squared;
      break;
    }
    module->launch("newton_weight_design", ceildiv(int64_t(n) * d), 256, 0, stream,
                   design.const_data_ptr<Real>(), roots.const_data_ptr<Real>(),
                   weighted.data_ptr<Real>(), int64_t(n) * d, n);
    linalg::dense_gram(blas, weighted.const_data_ptr<Real>(), matrix.data_ptr<Real>(), n, d);
    module->launch("newton_ridge_scale", 1, 256, 0, stream, matrix.data_ptr<Real>(), Real(ridge),
                   scalars.data_ptr<double>(), d);
    receive(1);
    double scale = values[0];
    bool factored = false;
    for (double multiplier : damping) {
      check(cudaMemcpyAsync(lower.data_ptr(), matrix.const_data_ptr(),
                            int64_t(d) * d * sizeof(Real), cudaMemcpyDeviceToDevice, stream));
      if (multiplier)
        module->launch("newton_damping", features, 256, 0, stream, lower.data_ptr<Real>(),
                       Real(multiplier * scale), d);
      linalg::dense_factor(solver, lower.data_ptr<Real>(), d, work.data_ptr<Real>(), count,
                           info.data_ptr<int>());
      if (status_info() == 0) {
        maximum_damping = std::max(maximum_damping, multiplier * scale);
        factored = true;
        break;
      }
    }
    if (!factored) {
      terminal = Status::factorization_failed;
      break;
    }
    check(cudaMemcpyAsync(step.data_ptr(), gradient.const_data_ptr(), d * sizeof(Real),
                          cudaMemcpyDeviceToDevice, stream));
    linalg::dense_solve(solver, lower.const_data_ptr<Real>(), step.data_ptr<Real>(), d,
                        info.data_ptr<int>());
    TORCH_CHECK(status_info() == 0, "CUDA Newton triangular solve failed");
    module->launch("newton_dot", features, 256, 0, stream, gradient.const_data_ptr<Real>(),
                   step.const_data_ptr<Real>(), parts.data_ptr<double>(), d);
    reduce(parts, features, 0);
    receive(1);
    decrement = values[0];
    if (!std::isfinite(decrement) || decrement <= 0) {
      terminal = Status::invalid_direction;
      break;
    }
    bool accepted = false;
    double rate = 1;
    for (int backtrack = 0; backtrack < 40; ++backtrack, rate *= 0.5) {
      module->launch("newton_candidate", features, 256, 0, stream, weights.const_data_ptr<Real>(),
                     step.const_data_ptr<Real>(), candidate.data_ptr<Real>(),
                     change.data_ptr<Real>(), Real(rate), ridge, parts.data_ptr<double>(), d);
      reduce(parts, features, 0);
      linalg::dense_mv(blas, design.const_data_ptr<Real>(), change.const_data_ptr<Real>(),
                       delta_logits.data_ptr<Real>(), n, d, false);
      module->launch("newton_loss_difference", rows, 256, 0, stream, logits.const_data_ptr<Real>(),
                     delta_logits.const_data_ptr<Real>(), labels.const_data_ptr<Real>(),
                     loss_parts.data_ptr<double>(), n);
      reduce(loss_parts, rows, 1);
      receive(2);
      double difference = values[0] + values[1];
      if (std::isfinite(loss) && std::isfinite(difference) &&
          difference <= -1e-4 * rate * decrement) {
        std::swap(weights, candidate);
        ++completed;
        gradient_current = false;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      terminal = Status::line_search_failed;
      break;
    }
  }
  if (!gradient_current)
    derivatives();
  if (std::isfinite(squared) && squared <= tolerance * tolerance)
    terminal = Status::gradient_squared;
  auto diagnostics = at::empty({3}, doubles.device(at::kCPU));
  double* result = diagnostics.data_ptr<double>();
  result[0] = squared;
  result[1] = decrement;
  result[2] = maximum_damping;
  return {weights, diagnostics, int64_t(terminal), completed};
}
} // namespace

bool setup(const std::string& path) {
  check(cudaFree(nullptr));
  kernels(path);
  return true;
}
Result fit(const Tensor& design, const Tensor& labels, const std::string& ptx, double ridge,
           double tolerance, int64_t max_iter, int64_t limit) {
  TORCH_CHECK(design.is_cuda() && design.dim() == 2 && design.size(0) > 0 && design.size(1) > 1 &&
                  design.size(0) <= INT_MAX && design.size(1) <= INT_MAX && design.stride(0) == 1 &&
                  design.stride(1) == design.size(0) &&
                  (design.scalar_type() == at::kFloat || design.scalar_type() == at::kDouble),
              "CUDA Newton requires a nonempty column-major FP32/FP64 design[N,D], D>1");
  TORCH_CHECK(labels.is_cuda() && labels.device() == design.device() && labels.dim() == 1 &&
                  labels.is_contiguous() && labels.numel() == design.size(0) &&
                  labels.scalar_type() == design.scalar_type(),
              "CUDA Newton labels must be same-device/dtype contiguous [N]");
  TORCH_CHECK(std::isfinite(ridge) && ridge >= 0 && std::isfinite(tolerance) && tolerance >= 0 &&
                  max_iter > 0 && limit >= 0 &&
                  int64_t(design.size(0)) * design.size(1) <= int64_t(INT_MAX) * 256 &&
                  int64_t(design.size(1)) * design.size(1) <= int64_t(INT_MAX) * 256,
              "invalid CUDA Newton budget or scalar parameters");
  Result result;
  AT_DISPATCH_FLOATING_TYPES(design.scalar_type(), "ltm_newton_fit", [&] {
    result = fit_impl<scalar_t>(design, labels, ptx, ridge, tolerance, max_iter, limit);
  });
  return result;
}
} // namespace ltm::cuda::classifier::newton
