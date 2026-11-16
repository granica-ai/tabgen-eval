#include "cuda/fidelity/conditional.hpp"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(tabgen_eval_cuda, m) {
  m.def("dense_tv_batches(Tensor a, Tensor b, Tensor pairs, Tensor levels, "
        "Tensor prototype, int grid, int batch, int cond, int[] histogram, "
        "int[] reduction) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(tabgen_eval_cuda, CUDA, m) {
  m.impl("dense_tv_batches", ltm::cuda::fidelity::dense_tv_batches);
}
