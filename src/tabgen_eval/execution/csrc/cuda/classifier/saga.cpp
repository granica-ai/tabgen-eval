// C++ owns CUDA minibatch/epoch orchestration. No ATen
// arithmetic, sorting or reductions are dispatched in the update loop.
#include "cuda/runtime/ptx.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

#include <climits>
#include <cmath>
#include <initializer_list>
#include <utility>

#include "cuda/classifier/packets.hpp"
#include "cuda/classifier/saga.hpp"
#include "cuda/classifier/saga_inverse.hpp"
#include <ATen/Dispatch.h>

namespace ltm::cuda::classifier::saga {
using at::Tensor;
namespace {
void check_runtime(cudaError_t e) {
  runtime::check_runtime(e, "native SAGA CUDA error: ");
}
void check_driver(CUresult e) {
  runtime::check_driver(e, "native SAGA driver error: ");
}
std::shared_ptr<runtime::KernelGroup> kernels(const std::string& path) {
  return runtime::kernels(path, "saga",
                          {"gather_numbers", "scale_categories", "reduce_sum", "logistic_delta",
                           "sort_categories", "category_bounds", "category_adjoint",
                           "gradient_average", "scale_vector", "update_weights", "gradient_initial",
                           "epoch_change", "epoch_reduce"},
                          "missing native PTX: ", "native SAGA driver error: ");
}
} // namespace

bool setup(const std::string& path) {
  check_runtime(cudaFree(nullptr));
  kernels(path);
  return true;
}
namespace {
int ceildiv(int n, int k) {
  return (n + k - 1) / k;
}
void floating(const Tensor& t, const Tensor& prototype) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == prototype.scalar_type() &&
                  (t.scalar_type() == at::kFloat || t.scalar_type() == at::kDouble) &&
                  t.device() == prototype.device(),
              "native SAGA prototype requires same-device FP32 or FP64 tensors");
}
template <class Real>
Result fit_impl(const std::vector<Tensor>& x, const std::string& ptx, double rate, double ridge,
                double tolerance, int64_t inverse_mode, bool stopping, bool averaged, int64_t limit,
                double initial_norm) {
  const FitPacketView inputs(x);
  const auto& numbers = inputs.numbers;
  floating(numbers, numbers);
  c10::cuda::CUDAGuard guard(numbers.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto module = kernels(ptx);
  int n = numbers.size(0), pn = numbers.size(1), pc = inputs.categories.size(1),
      k = inputs.category_centers.numel(), d = pn + k + 1;
  TORCH_CHECK(numbers.dim() == 2 && numbers.is_contiguous() && numbers.size(0) <= INT32_MAX,
              "numeric design must be contiguous and fit int32 row indices");
  TORCH_CHECK(inputs.categories.is_cuda() && inputs.categories.scalar_type() == at::kLong &&
                  inputs.categories.is_contiguous() && inputs.categories.size(0) == n &&
                  inputs.categories.device() == numbers.device(),
              "invalid categorical identities");
  for (const auto& [i, tensor] :
       std::initializer_list<std::pair<int, const Tensor*>>{{2, &inputs.category_centers},
                                                            {3, &inputs.category_scales},
                                                            {4, &inputs.labels},
                                                            {6, &inputs.inverse_root},
                                                            {8, &inputs.tau},
                                                            {10, &inputs.ratios},
                                                            {12, &inputs.initial_weights},
                                                            {13, &inputs.initial_table},
                                                            {14, &inputs.initial_average}}) {
    floating(*tensor, numbers);
    TORCH_CHECK(tensor->is_contiguous(), "invalid contiguous input", i);
  }
  TORCH_CHECK(inputs.category_scales.numel() == k && inputs.labels.numel() == n &&
                  inputs.initial_weights.numel() == d && inputs.initial_table.numel() == n &&
                  inputs.initial_average.numel() == d,
              "native SAGA input dimensions disagree");
  const auto& schedule = inputs.schedule;
  TORCH_CHECK(schedule.is_cuda() && schedule.scalar_type() == at::kInt &&
                  schedule.is_contiguous() && schedule.device() == numbers.device() &&
                  schedule.dim() == 3,
              "native SAGA schedule must be contiguous CUDA int32[E,S,B]");
  int epochs = schedule.size(0), steps = schedule.size(1), b = schedule.size(2), sort_n = 1;
  TORCH_CHECK(b > 0 && b <= 4096 && b <= n && k < INT32_MAX && epochs > 0 && steps > 0,
              "unsupported minibatch/domain dimensions");
  TORCH_CHECK(std::isfinite(rate) && rate > 0 && std::isfinite(ridge) && ridge >= 0 &&
                  std::isfinite(tolerance) && tolerance >= 0 && (!std::isnan(initial_norm)) &&
                  (initial_norm >= 0 || initial_norm == -1),
              "invalid native SAGA scalar parameters");
  while (sort_n < b)
    sort_n <<= 1;
  TORCH_CHECK(int64_t(b) * pn <= INT32_MAX && int64_t(pc) * sort_n <= INT32_MAX &&
                  int64_t(k) * 32 <= INT32_MAX,
              "native launch dimensions exceed int32 capacity");
  int rank = inverse_mode ? inputs.ratios.numel() : 0;
  TORCH_CHECK(inverse_mode >= 0 && inverse_mode <= 2,
              "inverse_mode must be0(identity),1(ORMQR),2(basis)");
  if (inverse_mode) {
    TORCH_CHECK(inputs.inverse_root.numel() == d && rank > 0 && rank <= d,
                "invalid factor diagonal/rank");
  }
  if (inverse_mode == 1) {
    for (const auto* tensor : {&inputs.reflectors, &inputs.rotations}) {
      floating(*tensor, numbers);
      TORCH_CHECK(tensor->stride(0) == 1 &&
                      (tensor->stride(1) == tensor->size(0) || tensor->size(1) == 1),
                  "factor matrices must be column-major");
    }
    TORCH_CHECK(inputs.reflectors.size(0) == d && inputs.reflectors.size(1) == rank &&
                    inputs.tau.numel() == rank && inputs.rotations.size(0) == rank &&
                    inputs.rotations.size(1) == rank,
                "invalid Householder factor dimensions");
  }
  if (inverse_mode == 2) {
    floating(inputs.basis, numbers);
    TORCH_CHECK(inputs.basis.dim() == 2 && inputs.basis.size(0) == d &&
                    inputs.basis.size(1) == rank && inputs.basis.stride(0) == 1 &&
                    (inputs.basis.stride(1) == d || rank == 1),
                "basis must be column-major[D,R]");
  }
  int blocks = ceildiv(d, 256), cat_blocks = ceildiv(k, 256), delta_blocks = ceildiv(b, 256);
  int shared = sort_n * 12, max_shared = 0, device = 0;
  check_runtime(cudaGetDevice(&device));
  check_runtime(
      cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(shared <= max_shared, "compact categorical sorting exceeds shared-memory allowance");
  if (pc)
    check_driver(cuFuncSetAttribute(module->function("sort_categories"),
                                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
  auto opts = numbers.options();
  auto ints = opts.dtype(at::kInt);
  auto keys_opts = opts.dtype(at::kLong);
  // Caller checks retained factor+schedule before copying them to the GPU;
  // this guards all additional fixed scratch, including the cached ORMQR work.
  int64_t basic = int64_t(sizeof(Real)) * (10LL * d + 2LL * n + int64_t(b) * pn + 2LL * b + k +
                                           cat_blocks + delta_blocks + 8 + 2LL * rank) +
                  12LL * pc * sort_n + 8LL * k + 8LL * (blocks * 4 + 1) + 48LL * epochs + 56;
  TORCH_CHECK_WITH(OutOfMemoryError, basic + schedule.nbytes() <= limit,
                   "native SAGA buffers exceed model workspace");
  auto weights = at::empty({d}, opts), table = at::empty({n}, opts), average = at::empty({d}, opts);
  auto summed = at::empty({d}, opts), previous = at::empty({d}, opts),
       gradient = at::empty({d}, opts), correction = at::empty({d}, opts);
  auto batch = at::empty({b, pn}, opts), logits = at::empty({b}, opts),
       delta = at::empty({b}, opts);
  auto scaled_cats = at::empty({std::max(1, k)}, opts),
       center_parts = at::empty({std::max(1, cat_blocks)}, opts);
  auto delta_parts = at::empty({delta_blocks}, opts), baseline = at::empty({1}, opts),
       total = at::empty({1}, opts);
  auto sorted_keys = at::empty({std::max(1, pc * sort_n)}, keys_opts),
       sorted_values = at::empty({std::max(1, pc * sort_n)}, ints);
  auto begin = at::empty({std::max(1, k)}, ints), end = at::empty({std::max(1, k)}, ints);
  auto history_opts = opts.dtype(at::kDouble);
  auto epoch_parts = at::empty({blocks * 4 + 1}, history_opts),
       epoch_out = at::empty({epoch_history::width}, history_opts),
       initial = at::full({1}, initial_norm, history_opts);
  auto history =
      at::empty({epochs, epoch_history::width}, history_opts.device(at::kCPU).pinned_memory(true));
  check_runtime(cudaMemcpyAsync(weights.data_ptr(), inputs.initial_weights.const_data_ptr(),
                                d * sizeof(Real), cudaMemcpyDeviceToDevice, stream));
  check_runtime(cudaMemcpyAsync(previous.data_ptr(), inputs.initial_weights.const_data_ptr(),
                                d * sizeof(Real), cudaMemcpyDeviceToDevice, stream));
  check_runtime(cudaMemcpyAsync(table.data_ptr(), inputs.initial_table.const_data_ptr(),
                                n * sizeof(Real), cudaMemcpyDeviceToDevice, stream));
  check_runtime(cudaMemcpyAsync(average.data_ptr(), inputs.initial_average.const_data_ptr(),
                                d * sizeof(Real), cudaMemcpyDeviceToDevice, stream));
  check_runtime(cudaMemsetAsync(summed.data_ptr(), 0, d * sizeof(Real), stream));
  auto coordinates = at::empty({d}, opts), projection = at::empty({std::max(1, rank)}, opts);
  auto rotated = at::empty({std::max(1, rank)}, opts), step_out = at::empty({d}, opts);
  auto info = at::empty({1}, ints);
  check_runtime(cudaMemsetAsync(info.data_ptr(), 0, 4, stream));
  auto blas = at::cuda::getCurrentCUDABlasHandle();
  auto solver = at::cuda::getCurrentCUDASolverDnHandle();
  inverse::BlasScope modes(blas);
  inverse::FactorView<Real> factor{d,
                                   rank,
                                   inputs.inverse_root.const_data_ptr<Real>(),
                                   inputs.reflectors.const_data_ptr<Real>(),
                                   inputs.tau.const_data_ptr<Real>(),
                                   inputs.rotations.const_data_ptr<Real>(),
                                   inputs.ratios.const_data_ptr<Real>()};
  int work_count =
      inverse_mode == 1 ? inverse::workspace_size(solver, factor, coordinates.data_ptr<Real>()) : 0;
  TORCH_CHECK_WITH(OutOfMemoryError,
                   basic + schedule.nbytes() + int64_t(work_count) * sizeof(Real) <= limit,
                   "cached ORMQR workspace exceeds model allowance");
  auto work = at::empty({std::max(1, work_count)}, opts);
  inverse::ScratchView<Real> scratch{coordinates.data_ptr<Real>(), projection.data_ptr<Real>(),
                                     rotated.data_ptr<Real>(),     work.data_ptr<Real>(),
                                     step_out.data_ptr<Real>(),    work_count,
                                     info.data_ptr<int>()};
  const Real one = 1, zero = 0;
  int completed = 0, updates = 0;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (int iteration = 0; iteration < steps; ++iteration) {
      const int* rows = schedule.const_data_ptr<int>() + (int64_t(epoch) * steps + iteration) * b;
      module->launch("gather_numbers", ceildiv(b * pn, 256), 256, 0, stream,
                     numbers.const_data_ptr<Real>(), rows, batch.data_ptr<Real>(), b, pn);
      if (pn)
        inverse::check_blas(inverse::gemv<Real>(
            blas, CUBLAS_OP_T, pn, b, &one, batch.const_data_ptr<Real>(), pn,
            weights.const_data_ptr<Real>(), 1, &zero, logits.data_ptr<Real>(), 1));
      else
        check_runtime(cudaMemsetAsync(logits.data_ptr(), 0, b * sizeof(Real), stream));
      if (k) {
        module->launch("scale_categories", cat_blocks, 256, 0, stream,
                       weights.const_data_ptr<Real>(),
                       inputs.category_centers.const_data_ptr<Real>(),
                       inputs.category_scales.const_data_ptr<Real>(), scaled_cats.data_ptr<Real>(),
                       center_parts.data_ptr<Real>(), k, pn);
        module->launch("reduce_sum", 1, 256, 0, stream, center_parts.const_data_ptr<Real>(),
                       baseline.data_ptr<Real>(), cat_blocks);
      } else
        check_runtime(cudaMemsetAsync(baseline.data_ptr(), 0, sizeof(Real), stream));
      module->launch("logistic_delta", delta_blocks, 256, 0, stream, logits.const_data_ptr<Real>(),
                     inputs.categories.const_data_ptr<int64_t>(), rows,
                     scaled_cats.const_data_ptr<Real>(), baseline.const_data_ptr<Real>(),
                     weights.const_data_ptr<Real>(), inputs.labels.const_data_ptr<Real>(),
                     table.data_ptr<Real>(), delta.data_ptr<Real>(), delta_parts.data_ptr<Real>(),
                     b, pc, d);
      module->launch("reduce_sum", 1, 256, 0, stream, delta_parts.const_data_ptr<Real>(),
                     total.data_ptr<Real>(), delta_blocks);
      if (pn)
        inverse::check_blas(inverse::gemv<Real>(
            blas, CUBLAS_OP_N, pn, b, &one, batch.const_data_ptr<Real>(), pn,
            delta.const_data_ptr<Real>(), 1, &zero, correction.data_ptr<Real>(), 1));
      if (k) {
        check_runtime(cudaMemsetAsync(begin.data_ptr(), 0xff, k * 4, stream));
        check_runtime(cudaMemsetAsync(end.data_ptr(), 0xff, k * 4, stream));
        module->launch(
            "sort_categories", pc, 256, shared, stream, inputs.categories.const_data_ptr<int64_t>(),
            rows, sorted_keys.data_ptr<int64_t>(), sorted_values.data_ptr<int>(), b, pc, sort_n);
        module->launch("category_bounds", ceildiv(pc * sort_n, 256), 256, 0, stream,
                       sorted_keys.const_data_ptr<int64_t>(), begin.data_ptr<int>(),
                       end.data_ptr<int>(), pc * sort_n, sort_n);
        module->launch("category_adjoint", ceildiv(k * 32, 256), 256, 0, stream,
                       sorted_values.const_data_ptr<int>(), begin.const_data_ptr<int>(),
                       end.const_data_ptr<int>(), delta.const_data_ptr<Real>(),
                       total.const_data_ptr<Real>(), inputs.category_centers.const_data_ptr<Real>(),
                       inputs.category_scales.const_data_ptr<Real>(), correction.data_ptr<Real>(),
                       k, pn);
      }
      module->launch("gradient_average", blocks, 256, 0, stream, correction.const_data_ptr<Real>(),
                     total.const_data_ptr<Real>(), average.data_ptr<Real>(),
                     weights.const_data_ptr<Real>(), gradient.data_ptr<Real>(), d, Real(ridge),
                     Real(b) / n);
      if (initial_norm < 0 && updates == 0)
        module->launch("gradient_initial", 1, 256, 0, stream, gradient.const_data_ptr<Real>(),
                       initial.data_ptr<double>(), d);
      const Real* update = gradient.const_data_ptr<Real>();
      int final_scale = 0;
      if (inverse_mode == 1) {
        inverse::apply(blas, solver, factor, gradient.const_data_ptr<Real>(), scratch);
        update = step_out.const_data_ptr<Real>();
      } else if (inverse_mode == 2) {
        module->launch("scale_vector", blocks, 256, 0, stream, gradient.const_data_ptr<Real>(),
                       inputs.inverse_root.const_data_ptr<Real>(), coordinates.data_ptr<Real>(), d);
        inverse::check_blas(inverse::gemv<Real>(
            blas, CUBLAS_OP_T, d, rank, &one, inputs.basis.const_data_ptr<Real>(), d,
            coordinates.const_data_ptr<Real>(), 1, &zero, projection.data_ptr<Real>(), 1));
        // Caller supplies ratios-1 for mode2, keeping the represented inverse unchanged.
        module->launch("scale_vector", ceildiv(rank, 256), 256, 0, stream,
                       projection.const_data_ptr<Real>(), inputs.ratios.const_data_ptr<Real>(),
                       rotated.data_ptr<Real>(), rank);
        inverse::check_blas(inverse::gemv<Real>(
            blas, CUBLAS_OP_N, d, rank, &one, inputs.basis.const_data_ptr<Real>(), d,
            rotated.const_data_ptr<Real>(), 1, &one, coordinates.data_ptr<Real>(), 1));
        update = coordinates.const_data_ptr<Real>();
        final_scale = 1;
      }
      module->launch("update_weights", blocks, 256, 0, stream, weights.data_ptr<Real>(), update,
                     inputs.inverse_root.const_data_ptr<Real>(), summed.data_ptr<Real>(), d,
                     Real(rate), int(averaged), final_scale);
      ++updates;
    }
    module->launch("epoch_change", blocks, 256, 0, stream, weights.const_data_ptr<Real>(),
                   previous.data_ptr<Real>(), gradient.const_data_ptr<Real>(),
                   epoch_parts.data_ptr<double>(), d);
    module->launch("epoch_reduce", 1, 256, 0, stream, epoch_parts.const_data_ptr<double>(),
                   initial.const_data_ptr<double>(), epoch_out.data_ptr<double>(), blocks);
    double* row = history.data_ptr<double>() + epoch * epoch_history::width;
    check_runtime(cudaMemcpyAsync(row, epoch_out.const_data_ptr(),
                                  epoch_history::width * sizeof(double), cudaMemcpyDeviceToHost,
                                  stream));
    check_runtime(cudaStreamSynchronize(stream));
    completed = epoch + 1;
    // Initial-gradient magnitude is retained only for diagnostics across chunks.
    if (row[epoch_history::finite] == 0 ||
        (stopping && row[epoch_history::online_gradient_infinity] <= tolerance))
      break;
  }
  int status = 0;
  check_runtime(cudaMemcpy(&status, info.const_data_ptr(), 4, cudaMemcpyDeviceToHost));
  TORCH_CHECK(status == 0, "cached ORMQR rejected arguments: ", status);
  // This is the existing variance-reduced minibatch gradient, including ridge
  // but BEFORE preconditioning and the final weight update. Returning its
  // owned buffer adds no gradient computation or allocation.
  return {weights, table, average, summed, history, completed, updates, gradient};
}
} // namespace

Result fit(const std::vector<Tensor>& x, const std::string& ptx, double rate, double ridge,
           double tolerance, int64_t inverse_mode, bool stopping, bool averaged, int64_t limit,
           double initial_norm) {
  TORCH_CHECK(!x.empty(), "SketchySAGA tensor packet is empty");
  Result result;
  AT_DISPATCH_FLOATING_TYPES(x[0].scalar_type(), "ltm_saga_fit", [&] {
    result = fit_impl<scalar_t>(x, ptx, rate, ridge, tolerance, inverse_mode, stopping, averaged,
                                limit, initial_norm);
  });
  return result;
}
} // namespace ltm::cuda::classifier::saga
