#pragma once

#include "cuda/runtime/errors.hpp"

namespace ltm::cuda::runtime {
// Borrowed PyTorch handles already carry the caller's stream. Restore only the
// modes changed here, including on exceptions; never destroy a borrowed handle.
class BlasScope {
public:
  cublasHandle_t handle;
  explicit BlasScope(cublasHandle_t value, const char* context = "native inverse cuBLAS error: ")
      : handle(value), context_(context) {
    check_blas(cublasGetMathMode(handle, &math_), context_);
    check_blas(cublasGetPointerMode(handle, &pointer_), context_);
    check_blas(cublasGetAtomicsMode(handle, &atomics_), context_);
    auto status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED);
    if (status != CUBLAS_STATUS_SUCCESS) {
      restore();
      check_blas(status, context_);
    }
  }
  BlasScope(const BlasScope&) = delete;
  BlasScope& operator=(const BlasScope&) = delete;
  ~BlasScope() {
    restore();
  }

private:
  void restore() noexcept {
    cublasSetMathMode(handle, math_);
    cublasSetPointerMode(handle, pointer_);
    cublasSetAtomicsMode(handle, atomics_);
  }
  cublasMath_t math_;
  cublasPointerMode_t pointer_;
  cublasAtomicsMode_t atomics_;
  const char* context_;
};
} // namespace ltm::cuda::runtime
