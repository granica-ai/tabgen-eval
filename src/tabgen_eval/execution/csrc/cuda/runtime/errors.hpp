#pragma once

#include <c10/util/Exception.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace ltm::cuda::runtime {
inline void check_runtime(cudaError_t status, const char* context) {
  TORCH_CHECK(status == cudaSuccess, context, cudaGetErrorString(status));
}
inline void check_driver(CUresult status, const char* context) {
  const char* message = nullptr;
  cuGetErrorString(status, &message);
  TORCH_CHECK(status == CUDA_SUCCESS, context, message ? message : "unknown");
}
inline void check_blas(cublasStatus_t status,
                       const char* context = "native inverse cuBLAS error: ") {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, context, int(status));
}
inline void check_solver(cusolverStatus_t status,
                         const char* context = "native inverse cuSOLVER error: ") {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, context, int(status));
}
} // namespace ltm::cuda::runtime
