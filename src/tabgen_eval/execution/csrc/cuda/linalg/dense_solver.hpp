#pragma once

#include "cuda/runtime/errors.hpp"
#include <type_traits>

namespace ltm::cuda::linalg {
// Typed library calls for column-major dense direction systems. Handles are
// borrowed from the caller; storage, workspace and status checks belong to the
// optimizer. No ATen numerical dispatch or implicit precision conversion occurs.
/** @brief Apply read-only column-major matrix[rows,columns] to a contiguous vector.
 * Write a same-precision vector on the borrowed handle stream; no allocation. */
template <class Real>
inline void dense_mv(cublasHandle_t handle, const Real* matrix, const Real* vector, Real* out,
                     int rows, int columns, bool transpose) {
  const Real one = 1, zero = 0;
  auto op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  if constexpr (std::is_same_v<Real, double>)
    runtime::check_blas(
        cublasDgemv(handle, op, rows, columns, &one, matrix, rows, vector, 1, &zero, out, 1));
  else
    runtime::check_blas(
        cublasSgemv(handle, op, rows, columns, &one, matrix, rows, vector, 1, &zero, out, 1));
}
/** @brief Write the lower triangle of matrix.T*matrix to out[columns,columns].
 * Inputs are read-only column-major FP32/FP64; no allocation or handle mutation. */
template <class Real>
inline void dense_gram(cublasHandle_t handle, const Real* matrix, Real* out, int rows,
                       int columns) {
  const Real one = 1, zero = 0;
  if constexpr (std::is_same_v<Real, double>)
    runtime::check_blas(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, columns, rows,
                                    &one, matrix, rows, &zero, out, columns));
  else
    runtime::check_blas(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, columns, rows,
                                    &one, matrix, rows, &zero, out, columns));
}
/** @brief Query POTRF scratch elements without reading or modifying factor. */
template <class Real>
inline int dense_factor_workspace(cusolverDnHandle_t handle, Real* factor, int width) {
  int count = 0;
  if constexpr (std::is_same_v<Real, double>)
    runtime::check_solver(
        cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, width, factor, width, &count));
  else
    runtime::check_solver(
        cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, width, factor, width, &count));
  TORCH_CHECK(count >= 0, "invalid cuSOLVER dense workspace size");
  return count;
}
/** @brief Factor the lower triangle in place with supplied same-precision work.
 * Write device int32 info; caller checks numerical status on its current stream. */
template <class Real>
inline void dense_factor(cusolverDnHandle_t handle, Real* factor, int width, Real* work, int count,
                         int* info) {
  if constexpr (std::is_same_v<Real, double>)
    runtime::check_solver(
        cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, width, factor, width, work, count, info));
  else
    runtime::check_solver(
        cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, width, factor, width, work, count, info));
}
/** @brief Solve a read-only lower Cholesky system, replacing contiguous rhs[width].
 * Write device int32 info; borrowed stream and handle state are unchanged. */
template <class Real>
inline void dense_solve(cusolverDnHandle_t handle, const Real* factor, Real* rhs, int width,
                        int* info) {
  if constexpr (std::is_same_v<Real, double>)
    runtime::check_solver(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, width, 1, factor, width,
                                           rhs, width, info));
  else
    runtime::check_solver(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, width, 1, factor, width,
                                           rhs, width, info));
}
} // namespace ltm::cuda::linalg
