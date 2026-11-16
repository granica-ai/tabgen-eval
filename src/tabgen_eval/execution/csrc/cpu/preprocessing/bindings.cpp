#include "cpu/preprocessing/operators.h"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(tabgen_eval_cpu, m) {
  m.def("preprocessing_count_missing(Tensor values, int threads=1) -> Tensor");
  m.def("preprocessing_assemble(Tensor numbers, Tensor codes, int feature_count, "
        "bool linear, int threads=1) -> Tensor");
  m.def("preprocessing_quantize(Tensor values, Tensor bits, ScalarType dtype, int threads=1) -> "
        "Tensor");
  m.def("preprocessing_fit_numeric(Tensor reference, Tensor candidate, "
        "bool linear, int threads=1) -> Tensor[]");
  m.def("preprocessing_apply_numeric(Tensor values, Tensor keep, Tensor indicators, Tensor impute, "
        "Tensor center, Tensor scale, bool linear, int threads=1) -> Tensor");
  m.def("preprocessing_fit_categories(Tensor reference, Tensor candidate, "
        "Tensor cardinalities, int threads=1) -> Tensor[]");
  m.def(
      "preprocessing_map_categories(Tensor codes, Tensor mapping, Tensor domains, Tensor features, "
      "bool linear, int threads=1) -> Tensor");
}
TORCH_LIBRARY_IMPL(tabgen_eval_cpu, CPU, m) {
  m.impl("preprocessing_count_missing", ltm::preprocessing::count_missing);
  m.impl("preprocessing_assemble", ltm::preprocessing::assemble);
  m.impl("preprocessing_quantize", ltm::preprocessing::quantize);
  m.impl("preprocessing_fit_numeric", ltm::preprocessing::fit_numeric);
  m.impl("preprocessing_apply_numeric", ltm::preprocessing::apply_numeric);
  m.impl("preprocessing_fit_categories", ltm::preprocessing::fit_categories);
  m.impl("preprocessing_map_categories", ltm::preprocessing::map_categories);
}
