#pragma once
// Small dense factorizations for native SketchySAGA preparation. PyTorch owns
// storage; cuSOLVER/cuBLAS execute on its current CUDA stream. Inputs are never
// mutated, and each returned matrix uses column-major storage.
#include "cuda/linalg/linear.hpp"
#include "cuda/runtime/blas_scope.hpp"
#include "cuda/runtime/errors.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ltm::cuda::linalg {
using at::Tensor;

inline void matrix(const Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(0) <= INT_MAX && x.size(1) <= INT_MAX &&
                  (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
              "native preparation requires nonempty FP32/FP64 CUDA matrices");
}
inline Tensor column_copy(const Tensor& x) {
  auto out = at::empty({x.size(1), x.size(0)}, x.options()).transpose(0, 1);
  out.copy_(x);
  return out;
}
inline void check_info(const Tensor& info, const char* operation) {
  const int value = info.item<int>();
  TORCH_CHECK(value >= 0, "native preparation ", operation, " rejected argument ", -value);
  TORCH_CHECK_LINALG(value == 0, "native preparation ", operation,
                     " numerical failure (info=", value, ")");
}

template <class Real, class... Args> inline cusolverStatus_t geqrf_size(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDgeqrf_bufferSize(args...);
  else
    return cusolverDnSgeqrf_bufferSize(args...);
}
template <class Real, class... Args> inline cusolverStatus_t geqrf(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDgeqrf(args...);
  else
    return cusolverDnSgeqrf(args...);
}
template <class Real, class... Args> inline cusolverStatus_t orgqr_size(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDorgqr_bufferSize(args...);
  else
    return cusolverDnSorgqr_bufferSize(args...);
}
template <class Real, class... Args> inline cusolverStatus_t orgqr(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDorgqr(args...);
  else
    return cusolverDnSorgqr(args...);
}
template <class Real, class... Args> inline cusolverStatus_t gesvd_size(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDgesvd_bufferSize(args...);
  else
    return cusolverDnSgesvd_bufferSize(args...);
}
template <class Real, class... Args> inline cusolverStatus_t gesvd(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDgesvd(args...);
  else
    return cusolverDnSgesvd(args...);
}
template <class Real, class... Args> inline cusolverStatus_t potrf_size(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDpotrf_bufferSize(args...);
  else
    return cusolverDnSpotrf_bufferSize(args...);
}
template <class Real, class... Args> inline cusolverStatus_t potrf(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDpotrf(args...);
  else
    return cusolverDnSpotrf(args...);
}

template <class Real>
inline int64_t workspace_bytes_impl(const Tensor& prototype, int d, int rank) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  // These legacy bufferSize entry points inspect dimension/type metadata only;
  // no matrix values or reflectors are consumed. A valid device pointer avoids
  // allocating a d-by-rank probe before the caller's workspace budget check.
  auto probe = at::empty({1}, prototype.options());
  Real* pointer = probe.data_ptr<Real>();
  int maximum = 0, count = 0;
  auto retain = [&]() {
    TORCH_CHECK(count >= 0, "native preparation invalid workspace size");
    maximum = std::max(maximum, count);
  };
  runtime::check_solver(geqrf_size<Real>(handle, d, rank, pointer, d, &count));
  retain();
  runtime::check_solver(orgqr_size<Real>(handle, d, rank, rank, pointer, d, pointer, &count));
  retain();
  runtime::check_solver(gesvd_size<Real>(handle, d, rank, &count));
  retain();
  runtime::check_solver(gesvd_size<Real>(handle, rank, rank, &count));
  retain();
  runtime::check_solver(
      potrf_size<Real>(handle, CUBLAS_FILL_MODE_LOWER, rank, pointer, rank, &count));
  retain();
  TORCH_CHECK(maximum >= 0, "native preparation invalid workspace size");
  return int64_t(maximum) * sizeof(Real);
}
// Maximum sequential library scratch allocation, excluding factor outputs and
// the SVD's rank-1 real rwork. The caller charges tensor storage independently.
inline int64_t workspace_bytes(const Tensor& prototype, int64_t d, int64_t rank) {
  TORCH_CHECK(
      prototype.is_cuda() &&
          (prototype.scalar_type() == at::kFloat || prototype.scalar_type() == at::kDouble) &&
          d > 0 && d <= INT_MAX && rank > 0 && rank <= d,
      "invalid native preparation workspace query");
  c10::cuda::CUDAGuard guard(prototype.device());
  return prototype.scalar_type() == at::kDouble
             ? workspace_bytes_impl<double>(prototype, int(d), int(rank))
             : workspace_bytes_impl<float>(prototype, int(d), int(rank));
}

template <class Real> inline std::tuple<Tensor, Tensor, Tensor> qr_impl(const Tensor& x) {
  const int m = x.size(0), n = x.size(1);
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto reflectors = column_copy(x);
  auto tau = at::empty({n}, x.options());
  auto info = at::empty({1}, x.options().dtype(at::kInt));
  int count = 0;
  runtime::check_solver(geqrf_size<Real>(handle, m, n, reflectors.data_ptr<Real>(), m, &count));
  auto scratch = at::empty({count}, x.options());
  runtime::check_solver(geqrf<Real>(handle, m, n, reflectors.data_ptr<Real>(), m,
                                    tau.data_ptr<Real>(), scratch.data_ptr<Real>(), count,
                                    info.data_ptr<int>()));
  check_info(info, "GEQRF");
  auto q = reflectors.clone(at::MemoryFormat::Preserve);
  runtime::check_solver(
      orgqr_size<Real>(handle, m, n, n, q.data_ptr<Real>(), m, tau.data_ptr<Real>(), &count));
  scratch = at::empty({count}, x.options());
  runtime::check_solver(orgqr<Real>(handle, m, n, n, q.data_ptr<Real>(), m, tau.data_ptr<Real>(),
                                    scratch.data_ptr<Real>(), count, info.data_ptr<int>()));
  check_info(info, "ORGQR");
  return {q, reflectors, tau};
}
// Reduced Q, packed QR reflectors (R in upper triangle), and Householder tau.
// Householder generation here is factorization, not repeated inverse application.
inline std::tuple<Tensor, Tensor, Tensor> qr(const Tensor& x) {
  matrix(x);
  TORCH_CHECK(x.size(0) >= x.size(1), "native QR requires rows >= columns");
  c10::cuda::CUDAGuard guard(x.device());
  return x.scalar_type() == at::kDouble ? qr_impl<double>(x) : qr_impl<float>(x);
}

template <class Real> inline std::tuple<Tensor, Tensor> svd_impl(const Tensor& x) {
  const int m = x.size(0), n = x.size(1);
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto a = column_copy(x);
  auto u = at::empty({n, m}, x.options()).transpose(0, 1);
  auto singular = at::empty({n}, x.options());
  auto unused_vt = at::empty({1}, x.options());
  auto rwork = at::empty({std::max(1, n - 1)}, x.options());
  auto info = at::empty({1}, x.options().dtype(at::kInt));
  int count = 0;
  runtime::check_solver(gesvd_size<Real>(handle, m, n, &count));
  auto scratch = at::empty({count}, x.options());
  runtime::check_solver(gesvd<Real>(handle, 'S', 'N', m, n, a.data_ptr<Real>(), m,
                                    singular.data_ptr<Real>(), u.data_ptr<Real>(), m,
                                    unused_vt.data_ptr<Real>(), 1, scratch.data_ptr<Real>(), count,
                                    rwork.data_ptr<Real>(), info.data_ptr<int>()));
  check_info(info, "GESVD");
  return {u, singular};
}
// Thin left singular vectors and singular values; right vectors are not formed.
inline std::tuple<Tensor, Tensor> svd(const Tensor& x) {
  matrix(x);
  TORCH_CHECK(x.size(0) >= x.size(1), "native SVD requires rows >= columns");
  c10::cuda::CUDAGuard guard(x.device());
  return x.scalar_type() == at::kDouble ? svd_impl<double>(x) : svd_impl<float>(x);
}

template <class Real> inline Tensor lower_factor_impl(const Tensor& x) {
  const int n = x.size(0);
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto factor = column_copy(x);
  auto info = at::empty({1}, x.options().dtype(at::kInt));
  int count = 0;
  runtime::check_solver(
      potrf_size<Real>(handle, CUBLAS_FILL_MODE_LOWER, n, factor.data_ptr<Real>(), n, &count));
  auto scratch = at::empty({count}, x.options());
  runtime::check_solver(potrf<Real>(handle, CUBLAS_FILL_MODE_LOWER, n, factor.data_ptr<Real>(), n,
                                    scratch.data_ptr<Real>(), count, info.data_ptr<int>()));
  check_info(info, "POTRF");
  return factor;
}
inline Tensor lower_factor(const Tensor& x) {
  matrix(x);
  TORCH_CHECK(x.size(0) == x.size(1), "native Cholesky requires square input");
  c10::cuda::CUDAGuard guard(x.device());
  return x.scalar_type() == at::kDouble ? lower_factor_impl<double>(x)
                                        : lower_factor_impl<float>(x);
}

// Solve lower * result = rhs for a matrix of right-hand sides.
inline Tensor triangular(const Tensor& lower, const Tensor& rhs) {
  matrix(lower);
  matrix(rhs);
  TORCH_CHECK(lower.size(0) == lower.size(1) && rhs.size(0) == lower.size(0) &&
                  rhs.device() == lower.device() && rhs.scalar_type() == lower.scalar_type(),
              "invalid native triangular solve inputs");
  c10::cuda::CUDAGuard guard(lower.device());
  auto factor = column_copy(lower);
  auto out = column_copy(rhs);
  const int m = out.size(0), n = out.size(1);
  runtime::BlasScope scope(at::cuda::getCurrentCUDABlasHandle());
  if (out.scalar_type() == at::kDouble) {
    const double one = 1;
    runtime::check_blas(cublasDtrsm(scope.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one,
                                    factor.const_data_ptr<double>(), m, out.data_ptr<double>(), m));
  } else {
    const float one = 1;
    runtime::check_blas(cublasStrsm(scope.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one,
                                    factor.const_data_ptr<float>(), m, out.data_ptr<float>(), m));
  }
  return out;
}

} // namespace ltm::cuda::linalg
