#include "cuda/classifier/preparation.hpp"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(tabgen_eval_cuda, m) {
  m.def("saga_preparation_policy(int n,bool enabled=True,int requested=4096)->int");
  m.impl("saga_preparation_policy", ltm::cuda::classifier::preparation::preparation_policy);
  m.def("saga_preparation_sample(int n,int count,int seed)->Tensor");
  m.impl("saga_preparation_sample", ltm::cuda::classifier::preparation::preparation_sample);
  m.def("saga_prepare(Tensor[] inputs,str ptx,float ridge,int rank,Generator? generator,int "
        "limit,bool sampled=True,int preparation_rows=4096,int batch=4096)->Tensor[]");
  m.impl("saga_prepare", ltm::cuda::classifier::preparation::prepare);
  m.def("saga_factor(Tensor diagonal,Tensor omega,Tensor images,float shift,int limit)->Tensor[]");
  m.impl("saga_factor", ltm::cuda::classifier::preparation::factor);
}
