#pragma once
// FP32/FP64 SketchySAGA: allocation-free inverse of D + L L^T, expressed
// in orthogonal Householder coordinates. Inputs are read-only. The caller
// owns stream-bound handles, the device guard, and all scratch buffers.
#include "cuda/runtime/blas_scope.hpp"
#include "cuda/runtime/errors.hpp"
#include <algorithm>
#include <c10/util/Exception.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <type_traits>

namespace ltm::cuda::classifier::inverse {
using runtime::BlasScope;
using runtime::check_blas;
using runtime::check_solver;
template <class Real, class... Args> inline cublasStatus_t gemv(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cublasDgemv(args...);
  else
    return cublasSgemv(args...);
}
template <class Real, class... Args> inline cublasStatus_t dgmm(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cublasDdgmm(args...);
  else
    return cublasSdgmm(args...);
}
template <class Real, class... Args> inline cusolverStatus_t ormqr(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDormqr(args...);
  else
    return cusolverDnSormqr(args...);
}
template <class Real, class... Args> inline cusolverStatus_t ormqr_workspace(Args... args) {
  if constexpr (std::is_same_v<Real, double>)
    return cusolverDnDormqr_bufferSize(args...);
  else
    return cusolverDnSormqr_bufferSize(args...);
}
template <class Real> struct FactorView {
  int d, rank;
  const Real *inverse_root, *reflectors, *tau, *rotations, *ratios;
};
template <class Real> struct ScratchView {
  Real *coordinates, *projection, *scaled, *workspace, *out;
  int workspace_count;
  int* info;
};
template <class Real>
inline int workspace_size(cusolverDnHandle_t solver, const FactorView<Real>& f,
                          const Real* coordinates) {
  int forward = 0, backward = 0;
  check_solver(ormqr_workspace<Real>(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, f.d, 1, f.rank,
                                     f.reflectors, f.d, f.tau, coordinates, f.d, &backward));
  check_solver(ormqr_workspace<Real>(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, f.d, 1, f.rank,
                                     f.reflectors, f.d, f.tau, coordinates, f.d, &forward));
  return std::max(forward, backward);
}
template <class Real>
inline void apply(cublasHandle_t blas, cusolverDnHandle_t solver, const FactorView<Real>& f,
                  const Real* rhs, const ScratchView<Real>& s) {
  const Real one = 1, zero = 0;
  // Diagonal scales are direct cuBLAS products, with no intermediate allocation.
  check_blas(
      dgmm<Real>(blas, CUBLAS_SIDE_LEFT, f.d, 1, rhs, f.d, f.inverse_root, 1, s.coordinates, f.d));
  check_solver(ormqr<Real>(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, f.d, 1, f.rank, f.reflectors, f.d,
                           f.tau, s.coordinates, f.d, s.workspace, s.workspace_count, s.info));
  check_blas(gemv<Real>(blas, CUBLAS_OP_T, f.rank, f.rank, &one, f.rotations, f.rank, s.coordinates,
                        1, &zero, s.projection, 1));
  check_blas(dgmm<Real>(blas, CUBLAS_SIDE_LEFT, f.rank, 1, s.projection, f.rank, f.ratios, 1,
                        s.scaled, f.rank));
  check_blas(gemv<Real>(blas, CUBLAS_OP_N, f.rank, f.rank, &one, f.rotations, f.rank, s.scaled, 1,
                        &zero, s.coordinates, 1));
  check_solver(ormqr<Real>(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, f.d, 1, f.rank, f.reflectors, f.d,
                           f.tau, s.coordinates, f.d, s.workspace, s.workspace_count, s.info));
  check_blas(dgmm<Real>(blas, CUBLAS_SIDE_LEFT, f.d, 1, s.coordinates, f.d, f.inverse_root, 1,
                        s.out, f.d));
}
} // namespace ltm::cuda::classifier::inverse
