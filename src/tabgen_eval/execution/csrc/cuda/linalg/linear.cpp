/** @file
 * @brief cuBLAS/cuSOLVER operators on PyTorch-owned CUDA buffers and current
 * streams.
 */
// CUDA linear algebra on PyTorch-owned storage and the caller's current stream.
#include "cuda/linalg/linear.hpp"
#include "cuda/runtime/blas_scope.hpp"
#include "cuda/runtime/errors.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace ltm::cuda::linalg {
using at::Tensor;
namespace {
void blas(cublasStatus_t status) {
  runtime::check_blas(status, "C2ST cuBLAS error: ");
}
void solver(cusolverStatus_t status) {
  runtime::check_solver(status, "C2ST cuSOLVER error: ");
}
void numeric(const Tensor& x) {
  TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
              "CUDA linear algebra requires FP32 or FP64 CUDA tensors");
}
void together(const Tensor& a, const Tensor& b) {
  numeric(a);
  numeric(b);
  TORCH_CHECK(a.device() == b.device() && a.scalar_type() == b.scalar_type(),
              "CUDA linear algebra inputs must share device and dtype");
}

} // namespace

// PyTorch owns stream-bound, per-thread/device library handles. Save every
// setting we change, including on exceptions. FP32 products request pedantic
// FP32 arithmetic regardless of ambient TF32 or autocast settings.
/** @brief Apply cuBLAS GEMV to CUDA float/double [N,D] row- or column-major x
 * and contiguous matching vector v; return [N] or [D] on the same dtype/device.
 * Inputs are read-only and caller-stream/handle math state is preserved. */
Tensor mv(const Tensor& x, const Tensor& v, bool transpose) {
  together(x, v);
  TORCH_CHECK(x.dim() == 2 && v.dim() == 1 && v.is_contiguous() &&
                  v.numel() == x.size(transpose ? 0 : 1) && x.size(0) <= INT32_MAX &&
                  x.size(1) <= INT32_MAX &&
                  (x.is_contiguous() || (x.stride(0) == 1 && x.stride(1) == x.size(0))),
              "invalid CUDA matrix-vector dimensions/layout");
  c10::cuda::CUDAGuard device(x.device());
  auto out = at::zeros({x.size(transpose ? 1 : 0)}, x.options());
  if (!x.numel())
    return out;
  runtime::BlasScope scope(at::cuda::getCurrentCUDABlasHandle(), "C2ST cuBLAS error: ");
  bool column = x.stride(0) == 1;
  int m = x.size(column ? 0 : 1), n = x.size(column ? 1 : 0);
  auto op = transpose == column ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (x.scalar_type() == at::kDouble) {
    double alpha = 1, beta = 0;
    blas(cublasDgemv(scope.handle, op, m, n, &alpha, x.const_data_ptr<double>(), m,
                     v.const_data_ptr<double>(), 1, &beta, out.data_ptr<double>(), 1));
  } else {
    float alpha = 1, beta = 0;
    blas(cublasSgemv(scope.handle, op, m, n, &alpha, x.const_data_ptr<float>(), m,
                     v.const_data_ptr<float>(), 1, &beta, out.data_ptr<float>(), 1));
  }
  return out;
}

/** @brief Multiply row/column-major CUDA matrices with declared FP32/FP64
 * arithmetic. Returns a column-major product and preserves all input buffers,
 * current device/stream, and ambient cuBLAS/TF32 policy. */
Tensor gemm(const Tensor& a, const Tensor& b) {
  together(a, b);
  auto matrix = [](const Tensor& x) {
    return x.dim() == 2 && x.size(0) <= INT32_MAX && x.size(1) <= INT32_MAX &&
           (x.is_contiguous() || (x.stride(0) == 1 && x.stride(1) == x.size(0)));
  };
  TORCH_CHECK(matrix(a) && matrix(b) && a.size(1) == b.size(0),
              "invalid CUDA matrix-matrix dimensions/layout");
  c10::cuda::CUDAGuard device(a.device());
  int m = a.size(0), n = b.size(1), k = a.size(1);
  auto out = at::zeros({n, m}, a.options()).transpose(0, 1);
  if (!m || !n || !k)
    return out;
  bool ac = a.stride(0) == 1, bc = b.stride(0) == 1;
  int lda = ac ? m : k, ldb = bc ? k : n;
  auto opa = ac ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto opb = bc ? CUBLAS_OP_N : CUBLAS_OP_T;
  runtime::BlasScope scope(at::cuda::getCurrentCUDABlasHandle(), "C2ST cuBLAS error: ");
  if (a.scalar_type() == at::kDouble) {
    double alpha = 1, beta = 0;
    blas(cublasDgemm(scope.handle, opa, opb, m, n, k, &alpha, a.const_data_ptr<double>(), lda,
                     b.const_data_ptr<double>(), ldb, &beta, out.data_ptr<double>(), m));
  } else {
    float alpha = 1, beta = 0;
    blas(cublasSgemm(scope.handle, opa, opb, m, n, k, &alpha, a.const_data_ptr<float>(), lda,
                     b.const_data_ptr<float>(), ldb, &beta, out.data_ptr<float>(), m));
  }
  return out;
}

/** @brief Return column-major CUDA [D,D] lower-triangle x.T*x via cuBLAS SYRK
 * for float/double [N,D]; upper triangle is zero. Inputs are read-only, FP32 is
 * pedantic IEEE, and handle modes/current device are restored. */
Tensor lower_gram(const Tensor& x) {
  numeric(x);
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 && x.size(0) <= INT32_MAX &&
                  x.size(1) <= INT32_MAX &&
                  (x.is_contiguous() || (x.stride(0) == 1 && x.stride(1) == x.size(0))),
              "invalid CUDA Gram layout");
  c10::cuda::CUDAGuard device(x.device());
  int n = x.size(0), d = x.size(1);
  auto out = at::zeros({d, d}, x.options()).transpose(0, 1);
  bool column = x.stride(0) == 1;
  runtime::BlasScope scope(at::cuda::getCurrentCUDABlasHandle(), "C2ST cuBLAS error: ");
  auto op = column ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = column ? n : d;
  if (x.scalar_type() == at::kDouble) {
    double alpha = 1, beta = 0;
    blas(cublasDsyrk(scope.handle, CUBLAS_FILL_MODE_LOWER, op, d, n, &alpha,
                     x.const_data_ptr<double>(), lda, &beta, out.data_ptr<double>(), d));
  } else {
    float alpha = 1, beta = 0;
    blas(cublasSsyrk(scope.handle, CUBLAS_FILL_MODE_LOWER, op, d, n, &alpha,
                     x.const_data_ptr<float>(), lda, &beta, out.data_ptr<float>(), d));
  }
  return out;
}

namespace {
void factor_input(const Tensor& x) {
  numeric(x);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == x.size(1) && x.size(0) > 0 && x.size(0) <= INT32_MAX &&
                  x.stride(0) == 1 && x.stride(1) == x.size(0),
              "cuSOLVER requires a square column-contiguous factor");
}
} // namespace

/** @brief Query cuSOLVER POTRF temporary bytes for positive square column-major
 * CUDA float/double input; performs no factorization or input mutation. */
int64_t workspace(const Tensor& x) {
  factor_input(x);
  c10::cuda::CUDAGuard device(x.device());
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  int n = x.size(0), count = 0;
  if (x.scalar_type() == at::kDouble)
    solver(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n,
                                       const_cast<double*>(x.const_data_ptr<double>()), n, &count));
  else
    solver(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n,
                                       const_cast<float*>(x.const_data_ptr<float>()), n, &count));
  return int64_t(count) * x.element_size();
}
/** @brief Factor a copied square column-major CUDA float/double lower triangle
 * via cuSOLVER. Returns factor and int32 [1] device info; caller checks
 * numerical status. Inputs remain read-only and scratch uses PyTorch
 * allocation/current stream. */
std::tuple<Tensor, Tensor> cholesky(const Tensor& x) {
  int64_t bytes = workspace(x);
  c10::cuda::CUDAGuard device(x.device());
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto out = x.clone(at::MemoryFormat::Preserve);
  auto scratch = at::empty({bytes / x.element_size()}, x.options());
  auto info = at::empty({1}, x.options().dtype(at::kInt));
  int n = x.size(0), count = scratch.numel();
  if (x.scalar_type() == at::kDouble)
    solver(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, out.data_ptr<double>(), n,
                            scratch.data_ptr<double>(), count, info.data_ptr<int>()));
  else
    solver(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, out.data_ptr<float>(), n,
                            scratch.data_ptr<float>(), count, info.data_ptr<int>()));
  // CUDA 11 cuSOLVER can return info=0 for NaN inputs. A successful
  // Cholesky factor must have a finite, strictly positive diagonal. Keep
  // this check on the caller's stream without a device-to-host sync.
  auto diagonal = out.diagonal();
  auto valid = at::isfinite(diagonal).logical_and(diagonal.gt(0)).all();
  info.masked_fill_(info.eq(0).logical_and(valid.logical_not()), 1);
  return {out, info};
}
/** @brief Solve one contiguous CUDA RHS [D] with a cached column-major Cholesky
 * factor [D,D]. Returns copied solution and int32 [1] device info; factor/RHS
 * remain read-only and cuSOLVER uses the caller current stream. */
std::tuple<Tensor, Tensor> solve(const Tensor& factor, const Tensor& rhs) {
  factor_input(factor);
  together(factor, rhs);
  TORCH_CHECK(rhs.dim() == 1 && rhs.is_contiguous() && rhs.numel() == factor.size(0),
              "cuSOLVER RHS must be a contiguous matching vector");
  c10::cuda::CUDAGuard device(factor.device());
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto out = rhs.clone();
  auto info = at::empty({1}, rhs.options().dtype(at::kInt));
  int n = factor.size(0);
  if (factor.scalar_type() == at::kDouble)
    solver(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1, factor.const_data_ptr<double>(),
                            n, out.data_ptr<double>(), n, info.data_ptr<int>()));
  else
    solver(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1, factor.const_data_ptr<float>(), n,
                            out.data_ptr<float>(), n, info.data_ptr<int>()));
  return {out, info};
}
} // namespace ltm::cuda::linalg
