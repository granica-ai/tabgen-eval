#include "cpu/classifier/operators.h"
#include "cpu/concentration/concentration.h"
#include "cpu/distance/nearest.h"
#include "cpu/distance/primitives.h"
#include "cpu/fidelity/primitives.h"
#include "cpu/runtime/draw_tasks.h"
#include <torch/library.h>

TORCH_LIBRARY(tabgen_eval_cpu, m) {
  m.def("local_concentration_counts(Tensor numeric, Tensor categorical, Tensor squared_thresholds, "
        "int workers, int scratch_bytes=67108864) -> Tensor");
  m.def("paired_concentration_distances(Tensor numeric, Tensor categorical, Tensor left, "
        "Tensor right, int workers=1) -> Tensor");
  m.def("distance_tile(Tensor a, Tensor b, Tensor ca, Tensor cb) -> Tensor");
  m.def("nearest_distances(Tensor reference, Tensor query, Tensor rc, Tensor qc, "
        "bool exclude_self, int threads, int scratch_bytes) -> Tensor");
  m.def("gram(Tensor a, Tensor b) -> Tensor");
  m.def("pair_counts(Tensor mask) -> Tensor");
  m.def("keys(Tensor codes, Tensor pairs, Tensor widths) -> Tensor");
  m.def("dense_counts(Tensor codes, Tensor pairs, Tensor widths, int grid) -> "
        "Tensor");
  m.def("sorted_tv(Tensor a, Tensor b, Tensor na, Tensor nb, Tensor prototype) "
        "-> Tensor");
  m.def("samples(Tensor rn, Tensor rc, Tensor sn, Tensor sc, Tensor rm, Tensor "
        "sm, Tensor rd, Tensor sd, int[] cardinalities, str "
        "metric, int k, int minimum, int bins, float rare, int threads, int components=15) -> "
        "Tensor");
  m.def("has_rare(Tensor numeric, Tensor categorical, int minimum, int bins, "
        "float rare) -> bool");
  m.def("fidelity_details(Tensor rn, Tensor rc, Tensor sn, Tensor sc, Tensor rm, Tensor sm, "
        "Tensor rd, Tensor sd, int[] cardinalities, str metric, int minimum, int bins, "
        "float rare, int threads, int components=7) -> Dict(str, Tensor)");
}
TORCH_LIBRARY_IMPL(tabgen_eval_cpu, CPU, m) {
  m.impl("local_concentration_counts", ltm::local_concentration_counts);
  m.impl("paired_concentration_distances", ltm::paired_concentration_distances);
  m.impl("distance_tile", ltm::distances);
  m.impl("nearest_distances", ltm::nearest_distances);
  m.impl("gram", ltm::gram);
  m.impl("pair_counts", ltm::pair_counts);
  m.impl("keys", ltm::keys);
  m.impl("dense_counts", ltm::dense);
  m.impl("sorted_tv", ltm::sorted_tv);
  m.impl("samples", ltm::samples);
  m.impl("has_rare", ltm::rare);
  m.impl("fidelity_details", ltm::fidelity_details);
}

TORCH_LIBRARY_FRAGMENT(tabgen_eval_cpu, m) {
  m.def("classifier_fit(Tensor x, Tensor y, str solver, float ridge, float "
        "tolerance, int max_iter, int direct_columns, float direct_work, int "
        "saga_max_epochs, int saga_batch_size, int saga_rank, "
        "int seed, int threads, bool "
        "saga_backoff=True, bool saga_sampled_preparation=True, int saga_preparation_rows=4096) -> "
        "(Tensor, Tensor, str, str)");
  m.def("classifier_derivatives(Tensor x, Tensor y, Tensor w, float ridge) -> "
        "(Tensor, Tensor, Tensor)");
  m.def("classifier(Tensor rn, Tensor rc, Tensor sn, Tensor sc, Tensor rm, "
        "Tensor sm, Tensor rd, Tensor sd, Tensor rf, Tensor sf, Tensor seeds, "
        "int folds, str solver, float ridge, float tolerance, int max_iter, "
        "int direct_columns, float direct_work, int "
        "saga_max_epochs, int saga_batch_size, int saga_rank, "
        "int memory, int workers, int fit_threads, "
        "bool saga_backoff=True, bool "
        "saga_sampled_preparation=True, int saga_preparation_rows=4096) -> (Tensor, "
        "Tensor, str[], str[], Tensor[][])");
}
TORCH_LIBRARY_IMPL(tabgen_eval_cpu, CPU, m) {
  m.impl("classifier_fit", ltm::classifier_fit);
  m.impl("classifier_derivatives", ltm::classifier_derivatives);
  m.impl("classifier", ltm::classifier);
}
