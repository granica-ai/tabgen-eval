// Native, memory-bounded full-training SketchySAGA preparation. No Python
// callbacks, materialized one-hot design, or per-direction dispatcher calls.
#include "cuda/linalg/linear.hpp"
#include "cuda/runtime/ptx.hpp"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

#include <memory>

#include "common/classifier/saga_preparation.hpp"
#include "common/classifier/saga_rate.hpp"
#include "common/classifier/saga_row_curvature.hpp"
#include "cuda/classifier/packets.hpp"
#include "cuda/classifier/preparation.hpp"
#include "cuda/classifier/preparation_factor.hpp"
#include "cuda/classifier/saga_inverse.hpp"

namespace ltm::cuda::classifier::preparation {
using at::Tensor;
namespace {
inline void check_runtime(cudaError_t e) {
  runtime::check_runtime(e, "native preparation CUDA: ");
}
std::shared_ptr<runtime::KernelGroup> kernels(const std::string& path) {
  return runtime::kernels(path, "preparation",
                          {"prep_diagonal_numeric", "prep_diagonal_finish",
                           "prep_diagonal_category", "prep_logits", "prep_adjoint",
                           "prep_finish_adjoint", "scale_categories", "reduce_sum", "scale_vector"},
                          "missing preparation PTX: ", "native preparation driver: ");
}
int ceildiv(int n, int size) {
  return (n + size - 1) / size;
}
template <class Real> Real norm(cublasHandle_t handle, int n, const Real* x) {
  Real value;
  if constexpr (std::is_same_v<Real, double>)
    inverse::check_blas(cublasDnrm2(handle, n, x, 1, &value));
  else
    inverse::check_blas(cublasSnrm2(handle, n, x, 1, &value));
  return value;
}
template <class Real> Real dot(cublasHandle_t handle, int n, const Real* x, const Real* y) {
  Real value;
  if constexpr (std::is_same_v<Real, double>)
    inverse::check_blas(cublasDdot(handle, n, x, 1, y, 1, &value));
  else
    inverse::check_blas(cublasSdot(handle, n, x, 1, y, 1, &value));
  return value;
}
template <class Real> void scale(cublasHandle_t handle, int n, Real value, Real* x) {
  if constexpr (std::is_same_v<Real, double>)
    inverse::check_blas(cublasDscal(handle, n, &value, x, 1));
  else
    inverse::check_blas(cublasSscal(handle, n, &value, x, 1));
}
template <class Real>
std::vector<Tensor> prepare_impl(const std::vector<Tensor>& original, const std::string& ptx,
                                 double ridge, int64_t requested_rank,
                                 const std::optional<at::Generator>& generator, int64_t limit,
                                 bool sampled, int64_t preparation_rows, int64_t batch) {
  const PreparationPacket full(original);
  auto x = full;
  c10::cuda::CUDAGuard guard(full.numbers.device());
  const int64_t global_n = full.numbers.size(0);
  TORCH_CHECK(preparation_rows > 0 && batch > 0, "invalid preparation rows/batch");
  const int64_t curvature_n = ltm_saga_preparation::rows(global_n, sampled, preparation_rows);
  const uint64_t seed = generator ? generator->current_seed() : 0;
  int64_t sample_hash = 0;
  if (curvature_n < global_n) {
    auto selected = ltm_saga_preparation::sample(global_n, curvature_n, seed);
    sample_hash = ltm_saga_preparation::hash(selected);
    auto host =
        at::from_blob(selected.data(), {curvature_n}, at::TensorOptions().dtype(at::kLong)).clone();
    auto rows = host.to(full.numbers.device());
    int64_t sampling_bytes = curvature_n * (full.numbers.size(1) * full.numbers.element_size() +
                                            full.categories.size(1) * 48 + 24) +
                             full.category_centers.numel() * 16;
    TORCH_CHECK_WITH(OutOfMemoryError, sampling_bytes <= limit,
                     "sampled curvature inputs exceed model workspace");
    limit -= sampling_bytes;
    x.numbers = full.numbers.index_select(0, rows);
    x.categories = full.categories.index_select(0, rows);
    // Retain the full categorical domain and training-fitted transforms.
    auto sorted = at::sort(x.categories.flatten(), true, 0, false);
    auto keys = std::get<0>(sorted), order = std::get<1>(sorted), keep = keys >= 0;
    auto valid = keys.masked_select(keep);
    x.category_rows =
        at::floor_divide(order.masked_select(keep), std::max<int64_t>(1, x.categories.size(1)))
            .contiguous();
    auto counts = at::bincount(valid, {}, full.category_centers.numel());
    x.category_offsets = at::cat({at::zeros({1}, counts.options()), counts.cumsum(0)});
  }
  const auto& numbers = x.numbers;
  TORCH_CHECK(numbers.is_cuda() && numbers.dim() == 2 && numbers.is_contiguous(),
              "preparation numeric design must be contiguous CUDA");
  int64_t nn = numbers.size(0), pnn = numbers.size(1), kk = x.category_centers.numel(),
          dd = pnn + kk + 1;
  TORCH_CHECK(nn > 0 && nn <= INT32_MAX && dd <= INT32_MAX && x.categories.size(1) <= INT32_MAX,
              "preparation dimensions exceed int32");
  int n = nn, pn = pnn, k = kk, d = dd, pc = x.categories.size(1),
      rank = std::min<int64_t>(requested_rank, d);
  TORCH_CHECK(rank > 0 && std::isfinite(ridge) && ridge >= 0, "invalid preparation rank/ridge");
  TORCH_CHECK(int64_t(d) * rank <= INT32_MAX && int64_t(k) * 32 <= INT32_MAX,
              "preparation factor/domain dimensions exceed int32");
  for (const auto* tensor : {&x.category_centers, &x.category_scales})
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == numbers.device() &&
                    tensor->scalar_type() == numbers.scalar_type() && tensor->is_contiguous() &&
                    tensor->numel() == k,
                "invalid category transform");
  for (const auto* tensor : {&x.categories, &x.category_rows, &x.category_offsets})
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == numbers.device() &&
                    tensor->scalar_type() == at::kLong && tensor->is_contiguous(),
                "invalid compact category indices");
  TORCH_CHECK(x.categories.dim() == 2 && x.categories.size(0) == n &&
                  x.category_offsets.numel() == k + 1,
              "invalid category dimensions");
  int tiles = ceildiv(n, 4096), catblocks = ceildiv(k, 256);
  // Includes all live retained factors, rank construction, bounded diagonal
  // tiles and library scratch. Input storage is charged by the fold caller.
  int64_t bound = linalg::workspace_bytes(numbers, d, rank) +
                  int64_t(sizeof(Real)) * (int64_t(d) * (16LL * rank + 80) + 4LL * n +
                                           int64_t(tiles) * pn + 16LL * rank * rank);
  TORCH_CHECK_WITH(OutOfMemoryError, bound <= limit,
                   "SketchySAGA native curvature exceeds model workspace: required ", bound,
                   " bytes, allowance ", limit);
  auto opts = numbers.options();
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto module = kernels(ptx);
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  inverse::BlasScope scope(handle);
  const Real one = 1, zero = 0;
  auto diagonal = at::empty({d}, opts), partials = at::empty({int64_t(tiles) * pn}, opts);
  module->launch("prep_diagonal_numeric", tiles * ceildiv(pn, 32), 256, 0, stream,
                 numbers.const_data_ptr<Real>(), partials.data_ptr<Real>(), n, pn, ceildiv(pn, 32));
  module->launch("prep_diagonal_finish", ceildiv(pn, 256), 256, 0, stream,
                 partials.const_data_ptr<Real>(), diagonal.data_ptr<Real>(), n, pn, tiles,
                 Real(ridge));
  diagonal.select(0, d - 1).fill_(n);
  module->launch(
      "prep_diagonal_category", catblocks, 256, 0, stream,
      x.category_offsets.const_data_ptr<int64_t>(), x.category_centers.const_data_ptr<Real>(),
      x.category_scales.const_data_ptr<Real>(), diagonal.data_ptr<Real>(), k, pn, Real(ridge));
  diagonal.select(0, d - 1).fill_(0.25);
  partials = Tensor();
  TORCH_CHECK_LINALG(at::isfinite(diagonal).all().item<bool>() && diagonal.gt(0).all().item<bool>(),
                     "SketchySAGA requires a positive finite curvature diagonal");
  auto logits = at::empty({n}, opts), scaled = at::empty({std::max(1, k)}, opts),
       parts = at::empty({std::max(1, catblocks)}, opts);
  auto baseline = at::empty({1}, opts), total = at::empty({1}, opts), image = at::empty({d}, opts);
  auto majorant = [&](const Tensor& v, const Tensor& out, double penalty) {
    if (pn)
      inverse::check_blas(
          inverse::gemv<Real>(handle, CUBLAS_OP_T, pn, n, &one, numbers.const_data_ptr<Real>(), pn,
                              v.const_data_ptr<Real>(), 1, &zero, logits.data_ptr<Real>(), 1));
    else
      check_runtime(cudaMemsetAsync(logits.data_ptr(), 0, int64_t(n) * sizeof(Real), stream));
    if (k) {
      module->launch("scale_categories", catblocks, 256, 0, stream, v.const_data_ptr<Real>(),
                     x.category_centers.const_data_ptr<Real>(),
                     x.category_scales.const_data_ptr<Real>(), scaled.data_ptr<Real>(),
                     parts.data_ptr<Real>(), k, pn);
      module->launch("reduce_sum", 1, 256, 0, stream, parts.const_data_ptr<Real>(),
                     baseline.data_ptr<Real>(), catblocks);
    } else
      check_runtime(cudaMemsetAsync(baseline.data_ptr(), 0, sizeof(Real), stream));
    module->launch("prep_logits", ceildiv(n, 256), 256, 0, stream, logits.data_ptr<Real>(),
                   x.categories.const_data_ptr<int64_t>(), scaled.const_data_ptr<Real>(),
                   baseline.const_data_ptr<Real>(), v.const_data_ptr<Real>(), n, pc, d);
    module->launch("reduce_sum", 1, 256, 0, stream, logits.const_data_ptr<Real>(),
                   total.data_ptr<Real>(), n);
    if (pn)
      inverse::check_blas(
          inverse::gemv<Real>(handle, CUBLAS_OP_N, pn, n, &one, numbers.const_data_ptr<Real>(), pn,
                              logits.const_data_ptr<Real>(), 1, &zero, out.data_ptr<Real>(), 1));
    module->launch("prep_adjoint", ceildiv(k * 32, 256), 256, 0, stream,
                   x.category_rows.const_data_ptr<int64_t>(),
                   x.category_offsets.const_data_ptr<int64_t>(), logits.const_data_ptr<Real>(),
                   total.const_data_ptr<Real>(), x.category_centers.const_data_ptr<Real>(),
                   x.category_scales.const_data_ptr<Real>(), v.const_data_ptr<Real>(),
                   out.data_ptr<Real>(), k, pn, Real(penalty));
    module->launch("prep_finish_adjoint", std::max(1, ceildiv(pn, 256)), 256, 0, stream,
                   out.data_ptr<Real>(), v.const_data_ptr<Real>(), total.const_data_ptr<Real>(), pn,
                   d, Real(penalty));
  };
  auto raw = at::randn({d, rank}, generator, opts);
  auto omega = std::get<0>(linalg::qr(raw));
  raw = Tensor();
  auto images = at::empty({rank, d}, opts).transpose(0, 1);
  for (int j = 0; j < rank; ++j) {
    auto v = omega.select(1, j).contiguous();
    majorant(v, images.select(1, j), ridge);
  }
  double epsilon = sizeof(Real) == 4 ? std::numeric_limits<float>::epsilon()
                                     : std::numeric_limits<double>::epsilon();
  double tiny =
      sizeof(Real) == 4 ? std::numeric_limits<float>::min() : std::numeric_limits<double>::min();
  double shift =
      std::max(double(norm<Real>(handle, d * rank, images.const_data_ptr<Real>())) * epsilon, tiny);
  auto factors = FactorPacket(nystrom_factor(diagonal, omega, images, shift));
  auto inverse_root = factors.inverse_root, reflectors = factors.reflectors, tau = factors.tau,
       rotations = factors.rotations, ratios = factors.ratios, basis = factors.basis,
       residual = factors.residual;
  int fallback = factors.fallback.item<int64_t>();
  images = Tensor();
  omega = Tensor();
  rank = ratios.numel();
  auto direction = at::randn({d}, generator, opts);
  scale<Real>(handle, d, Real(1) / norm<Real>(handle, d, direction.const_data_ptr<Real>()),
              direction.data_ptr<Real>());
  auto solved = at::empty({d}, opts), coordinates = at::empty({d}, opts),
       projection = at::empty({rank}, opts), rotated = at::empty({rank}, opts);
  auto ratios_delta = ratios - 1;
  bool stable = sizeof(Real) == 4;
  auto solver = at::cuda::getCurrentCUDASolverDnHandle();
  auto info = at::zeros({1}, opts.dtype(at::kInt));
  inverse::FactorView<Real> factor{d,
                                   rank,
                                   inverse_root.const_data_ptr<Real>(),
                                   reflectors.const_data_ptr<Real>(),
                                   tau.const_data_ptr<Real>(),
                                   rotations.const_data_ptr<Real>(),
                                   ratios.const_data_ptr<Real>()};
  int work_count =
      stable ? inverse::workspace_size(solver, factor, coordinates.const_data_ptr<Real>()) : 0;
  TORCH_CHECK_WITH(OutOfMemoryError, bound + int64_t(work_count) * sizeof(Real) <= limit,
                   "native curvature factor workspace exceeds model allowance");
  auto work = at::empty({std::max(1, work_count)}, opts);
  inverse::ScratchView<Real> scratch{coordinates.data_ptr<Real>(), projection.data_ptr<Real>(),
                                     rotated.data_ptr<Real>(),     work.data_ptr<Real>(),
                                     solved.data_ptr<Real>(),      work_count,
                                     info.data_ptr<int>()};
  double eigenvalue = 0;
  for (int iteration = 0; iteration < 30; ++iteration) {
    majorant(direction, image, 0);
    if (stable)
      inverse::apply(handle, solver, factor, image.const_data_ptr<Real>(), scratch);
    else {
      module->launch("scale_vector", ceildiv(d, 256), 256, 0, stream, image.const_data_ptr<Real>(),
                     inverse_root.const_data_ptr<Real>(), coordinates.data_ptr<Real>(), d);
      inverse::check_blas(inverse::gemv<Real>(
          handle, CUBLAS_OP_T, d, rank, &one, basis.const_data_ptr<Real>(), d,
          coordinates.const_data_ptr<Real>(), 1, &zero, projection.data_ptr<Real>(), 1));
      module->launch("scale_vector", ceildiv(rank, 256), 256, 0, stream,
                     projection.const_data_ptr<Real>(), ratios_delta.const_data_ptr<Real>(),
                     rotated.data_ptr<Real>(), rank);
      inverse::check_blas(inverse::gemv<Real>(
          handle, CUBLAS_OP_N, d, rank, &one, basis.const_data_ptr<Real>(), d,
          rotated.const_data_ptr<Real>(), 1, &one, coordinates.data_ptr<Real>(), 1));
      module->launch("scale_vector", ceildiv(d, 256), 256, 0, stream,
                     coordinates.const_data_ptr<Real>(), inverse_root.const_data_ptr<Real>(),
                     solved.data_ptr<Real>(), d);
    }
    eigenvalue =
        double(dot<Real>(handle, d, image.const_data_ptr<Real>(), solved.const_data_ptr<Real>())) /
        dot<Real>(handle, d, direction.const_data_ptr<Real>(), image.const_data_ptr<Real>());
    Real length = norm<Real>(handle, d, solved.const_data_ptr<Real>());
    check_runtime(cudaMemcpyAsync(direction.data_ptr(), solved.const_data_ptr(), d * sizeof(Real),
                                  cudaMemcpyDeviceToDevice, stream));
    scale<Real>(handle, d, Real(1) / length, direction.data_ptr<Real>());
  }
  TORCH_CHECK(info.item<int>() == 0, "native curvature ORMQR rejected arguments");
  TORCH_CHECK_LINALG(std::isfinite(eigenvalue) && eigenvalue > 0,
                     "SketchySAGA curvature estimate is not finite and positive");
  auto calibration_x = x;
  int64_t calibration_n = std::min<int64_t>(4096, curvature_n);
  int64_t calibration_bytes =
      int64_t(sizeof(Real)) * calibration_n * (2LL * pn + 4LL * rank + 2LL * pc + 16);
  TORCH_CHECK_WITH(OutOfMemoryError, bound + calibration_bytes <= limit,
                   "native curvature calibration exceeds model workspace");
  if (calibration_n < curvature_n) {
    auto selected = ltm_saga_preparation::sample(curvature_n, calibration_n, seed ^ 0xCA11);
    auto rows =
        at::from_blob(selected.data(), {calibration_n}, at::TensorOptions().dtype(at::kLong))
            .clone()
            .to(numbers.device());
    calibration_x.numbers = numbers.index_select(0, rows);
    calibration_x.categories = x.categories.index_select(0, rows);
  }
  auto row_curvatures = ltm::saga::calibration::row_curvature(
      calibration_x.numbers, calibration_x.categories, x.category_centers, x.category_scales,
      inverse_root, basis, ratios, linalg::gemm);
  auto calibration = ltm::saga::calibration::rate(
      global_n, batch, calibration_n, eigenvalue, row_curvatures.max().item<double>(),
      ltm::saga::calibration::ridge_bound(inverse_root, ridge));
  double rate = calibration.rate;
  eigenvalue = calibration.data_used;
  PreparationDiagnostics diagnostics{double(fallback),
                                     eigenvalue,
                                     rate,
                                     residual.min().item<double>(),
                                     ratios.min().item<double>(),
                                     double(global_n),
                                     double(curvature_n),
                                     double(curvature_n < global_n),
                                     double(sample_hash),
                                     calibration.data_raw,
                                     calibration.data_used,
                                     calibration.maximum_raw,
                                     calibration.maximum_used,
                                     calibration.ridge_upper,
                                     calibration.practical,
                                     double(calibration.rows),
                                     double(calibration.population),
                                     double(calibration.batch)};
  auto metadata = diagnostics.tensor();
  return {inverse_root, reflectors, tau, rotations, ratios, basis, metadata};
}
} // namespace

std::vector<Tensor> factor(const Tensor& diagonal, const Tensor& omega, const Tensor& images,
                           double shift, int64_t limit) {
  linalg::matrix(omega);
  c10::cuda::CUDAGuard guard(omega.device());
  int64_t d = omega.size(0), rank = omega.size(1);
  TORCH_CHECK(rank <= d, "native factor rank exceeds width");
  int64_t bound = linalg::workspace_bytes(omega, d, rank) +
                  omega.element_size() * (16 * d * rank + 80 * d + 16 * rank * rank);
  TORCH_CHECK_WITH(OutOfMemoryError, bound <= limit,
                   "native factor exceeds model workspace: required ", bound, " bytes, allowance ",
                   limit);
  return nystrom_factor(diagonal, omega, images, shift);
}

std::vector<Tensor> prepare(const std::vector<Tensor>& x, const std::string& ptx, double ridge,
                            int64_t rank, const std::optional<at::Generator>& generator,
                            int64_t limit, bool sampled, int64_t preparation_rows, int64_t batch) {
  TORCH_CHECK(!x.empty(), "empty native preparation packet");
  std::vector<Tensor> result;
  AT_DISPATCH_FLOATING_TYPES(x[0].scalar_type(), "ltm_native_preparation", [&] {
    result = prepare_impl<scalar_t>(x, ptx, ridge, rank, generator, limit, sampled,
                                    preparation_rows, batch);
  });
  return result;
}
int64_t preparation_policy(int64_t n, bool enabled, int64_t requested) {
  TORCH_CHECK(n > 0 && requested > 0, "invalid preparation row policy inputs");
  return ltm_saga_preparation::rows(n, enabled, requested);
}
at::Tensor preparation_sample(int64_t n, int64_t count, int64_t seed) {
  TORCH_CHECK(n > 0 && count > 0 && count <= n, "invalid preparation sample inputs");
  auto selected = ltm_saga_preparation::sample(n, count, uint64_t(seed));
  return at::from_blob(selected.data(), {count}, at::TensorOptions().dtype(at::kLong)).clone();
}
} // namespace ltm::cuda::classifier::preparation
