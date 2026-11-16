#include "cuda/classifier/newton.hpp"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(tabgen_eval_cuda, m) {
  m.def("newton_setup(str ptx)->bool");
  m.def("newton_fit(Tensor design,Tensor labels,str ptx,float ridge,float tolerance,int "
        "max_iter,int limit)->(Tensor,Tensor,int,int)");
  m.impl("newton_setup", ltm::cuda::classifier::newton::setup);
  m.impl("newton_fit", ltm::cuda::classifier::newton::fit);
}
