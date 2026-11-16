#include "cuda/classifier/saga.hpp"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(tabgen_eval_cuda, m) {
  m.def("saga_setup(str ptx)->bool");
  m.def("saga_fit(Tensor[] inputs,str ptx,float rate,float ridge,float tolerance,int "
        "inverse_mode,bool stopping,bool averaged,int limit,float "
        "initial_norm=-1)->(Tensor,Tensor,Tensor,Tensor,Tensor,int,int,Tensor)");
  m.impl("saga_setup", ltm::cuda::classifier::saga::setup);
  m.impl("saga_fit", ltm::cuda::classifier::saga::fit);
}
