#include "cuda/linalg/linear.hpp"
#include <torch/library.h>

TORCH_LIBRARY(tabgen_eval_cuda, m) {
  m.def("mv(Tensor x, Tensor v, bool transpose=False) -> Tensor");
  m.def("gemm(Tensor a, Tensor b) -> Tensor");
  m.def("lower_gram(Tensor x) -> Tensor");
  m.def("potrf_workspace(Tensor x) -> int");
  m.def("cholesky(Tensor x) -> (Tensor, Tensor)");
  m.def("solve(Tensor factor, Tensor rhs) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(tabgen_eval_cuda, CUDA, m) {
  m.impl("mv", ltm::cuda::linalg::mv);
  m.impl("gemm", ltm::cuda::linalg::gemm);
  m.impl("lower_gram", ltm::cuda::linalg::lower_gram);
  m.impl("potrf_workspace", ltm::cuda::linalg::workspace);
  m.impl("cholesky", ltm::cuda::linalg::cholesky);
  m.impl("solve", ltm::cuda::linalg::solve);
}
