#include "cpu/classifier/detail/saga.h"
#include "common/classifier/saga_preparation.hpp"
#include "cpu/runtime/tensor_validation.h"
#include "cpu/runtime/thread_budget.h"
#include <numeric>
#include <torch/library.h>
#include <unordered_set>

namespace {
using ltm::Tensor;
std::tuple<Tensor, Tensor, std::string>
saga_fit(const Tensor& numbers, const Tensor& ids, const Tensor& centers, const Tensor& scales,
         const Tensor& labels, double ridge, double tol, int64_t epochs, int64_t batch,
         int64_t rank, int64_t seed, int64_t memory, int64_t threads, bool backoff, bool sampled,
         int64_t preparation_rows) {
  TORCH_CHECK(threads > 0 && threads <= INT32_MAX, "invalid thread budget");
  ltm::ThreadBudget budget(threads);
  auto r = ltm::saga::fit({numbers, ids, centers, scales}, labels, ridge, tol, epochs, batch, rank,
                          uint64_t(seed), memory, backoff, sampled, preparation_rows);
  return {r.coefficients, ltm::saga::fit_diagnostics(r, numbers.options()), r.reason};
}
// Numerical-oracle entrypoint: prescribed factors and schedule use exactly the
// production chunk/guard loop. It cannot override public solver settings.
std::tuple<std::vector<Tensor>, Tensor, std::string>
saga_guarded_updates(const std::vector<Tensor>& x, const Tensor& schedule, double ridge,
                     double rate, double tolerance, int64_t seed, int64_t epochs, bool backoff,
                     int64_t threads) {
  TORCH_CHECK(x.size() == 11 && threads > 0 && threads <= INT32_MAX && epochs > 0 &&
                  schedule.device().is_cpu() && schedule.scalar_type() == at::kLong &&
                  schedule.dim() == 3 && schedule.is_contiguous() &&
                  (schedule.size(0) == 0 || schedule.size(0) >= epochs) && schedule.size(2) > 0 &&
                  std::isfinite(tolerance) && tolerance > 0,
              "invalid guarded update packet/options");
  ltm::ThreadBudget budget(threads);
  ltm::saga::Design design{x[0], x[1], x[2], x[3]};
  TORCH_CHECK(schedule.size(1) == (design.n() + schedule.size(2) - 1) / schedule.size(2),
              "invalid scheduled epoch length");
  // Reuse the production update oracle's state/factor and unique-row checks,
  // but perform no updates during validation.
  auto flat = schedule.reshape({-1, schedule.size(2)});
  ltm::saga::detail::prescribed_updates(design, x[4], flat.narrow(0, 0, 0), x[5], x[6], x[7], x[8],
                                        x[9], x[10], ridge, rate);
  std::unordered_set<int64_t> unique;
  for (int64_t i = 0; i < flat.size(0); ++i) {
    unique.clear();
    for (int64_t j = 0; j < flat.size(1); ++j) {
      auto row = flat.const_data_ptr<int64_t>()[i * flat.size(1) + j];
      TORCH_CHECK(row >= 0 && row < design.n() && unique.insert(row).second,
                  "minibatch rows must be unique and in bounds");
    }
  }
  TORCH_CHECK((x[5] == 0).all().item<bool>() && (x[6] == 0).all().item<bool>() &&
                  (x[7] == 0).all().item<bool>() && ((x[4] == 0) | (x[4] == 1)).all().item<bool>(),
              "guarded fitting must start at the zero state with binary labels");
  ltm::saga::detail::State state{x[5].clone(), x[6].clone(), x[7].clone(), at::zeros_like(x[5]),
                                 x[5].clone()};
  std::optional<ltm::saga::detail::Inverse> inverse;
  if (x[8].numel()) {
    inverse.emplace();
    inverse->root = x[8];
    inverse->basis = x[9];
    inverse->ratios = x[10];
  }
  auto result = ltm::saga::detail::run(design, x[4], ridge, tolerance, epochs, schedule.size(2),
                                       uint64_t(seed), rate, inverse ? &*inverse : nullptr, backoff,
                                       state, schedule.size(0) ? schedule : Tensor{});
  return {{state.weights, state.table, state.average, state.gradient, state.previous},
          ltm::saga::fit_diagnostics(result, x[0].options()),
          result.reason};
}
std::vector<Tensor> saga_updates(const std::vector<Tensor>& x, const Tensor& schedule, double ridge,
                                 double rate, int64_t threads) {
  TORCH_CHECK(x.size() == 11 && threads > 0 && threads <= INT32_MAX,
              "invalid update packet/thread budget");
  ltm::ThreadBudget budget(threads);
  return ltm::saga::detail::prescribed_updates({x[0], x[1], x[2], x[3]}, x[4], schedule, x[5], x[6],
                                               x[7], x[8], x[9], x[10], ridge, rate);
}

Tensor saga_mv(const Tensor& numbers, const Tensor& ids, const Tensor& centers,
               const Tensor& scales, const Tensor& weights) {
  ltm::saga::Design x{numbers, ids, centers, scales};
  ltm::saga::detail::valid(x);
  TORCH_CHECK(weights.device().is_cpu() && weights.scalar_type() == numbers.scalar_type() &&
                  weights.dim() == 1 && weights.numel() == x.d() && weights.is_contiguous() &&
                  at::isfinite(weights).all().item<bool>(),
              "invalid scoring coefficients");
  return ltm::saga::mv(x, weights);
}
int64_t saga_preparation_rows(int64_t n, bool enabled, int64_t requested) {
  TORCH_CHECK(n > 0 && requested > 0, "invalid preparation row counts");
  return ltm_saga_preparation::rows(n, enabled, requested);
}
Tensor saga_preparation_sample(int64_t n, int64_t requested, int64_t seed) {
  TORCH_CHECK(n > 0 && requested > 0 && requested <= n, "invalid preparation sample dimensions");
  auto selected = ltm_saga_preparation::sample(n, requested, uint64_t(seed));
  return at::from_blob(selected.data(), {requested}, at::TensorOptions().dtype(at::kLong)).clone();
}
std::vector<Tensor> saga_prepare(const Tensor& rn, const Tensor& rc, const Tensor& sn,
                                 const Tensor& sc, const Tensor& rf, const Tensor& sf, int64_t fold,
                                 int64_t memory) {
  ltm::numbers(rn);
  ltm::numbers(sn);
  TORCH_CHECK(rn.scalar_type() == sn.scalar_type(), "numeric pool dtypes must match");
  ltm::matrix(rc, at::kLong);
  ltm::matrix(sc, at::kLong);
  TORCH_CHECK(rn.size(0) == rc.size(0) && sn.size(0) == sc.size(0) && rn.size(1) == sn.size(1) &&
                  rc.size(1) == sc.size(1) && rf.device().is_cpu() && sf.device().is_cpu() &&
                  rf.scalar_type() == at::kLong && sf.scalar_type() == at::kLong &&
                  rf.is_contiguous() && sf.is_contiguous() && rf.numel() == rn.size(0) &&
                  sf.numel() == sn.size(0) && memory > 0,
              "invalid compact preparation inputs");
  auto mask = at::ones({rn.size(0)}, rn.options().dtype(at::kBool));
  auto smask = at::ones({sn.size(0)}, sn.options().dtype(at::kBool));
  std::vector<int64_t> r(rn.size(0)), s(sn.size(0));
  std::iota(r.begin(), r.end(), 0);
  std::iota(s.begin(), s.end(), 0);
  return AT_DISPATCH_FLOATING_TYPES(rn.scalar_type(), "cpu_saga_prepare", [&]() {
    auto a = ltm::gather<scalar_t>(rn, rc, mask, r.data(), r.size()),
         b = ltm::gather<scalar_t>(sn, sc, smask, s.data(), s.size());
    auto x = ltm::saga::prepare(a, b, rf.const_data_ptr<int64_t>(), sf.const_data_ptr<int64_t>(),
                                fold, memory);
    if (!x)
      return std::vector<Tensor>{};
    return std::vector<Tensor>{x->train.numbers,   x->train.identities, x->test.numbers,
                               x->test.identities, x->train.centers,    x->train.scales,
                               x->labels,          x->scoring};
  });
}
} // namespace
TORCH_LIBRARY_FRAGMENT(tabgen_eval_cpu, m) {
  m.def("classifier_saga_updates(Tensor[] inputs, Tensor schedule, float "
        "ridge, float rate, int threads) -> Tensor[]");
  m.def("classifier_saga_fit(Tensor numbers, Tensor identities, Tensor "
        "centers, Tensor scales, Tensor labels, float ridge, float tolerance, "
        "int epochs, int batch_size, int rank, int seed, int memory, int "
        "threads, bool backoff=True, "
        "bool sampled=True, int preparation_rows=4096) -> (Tensor, Tensor, str)");
  m.def("classifier_saga_guarded_updates(Tensor[] inputs, Tensor schedule, float "
        "ridge, float rate, float tolerance, int seed, "
        "int epochs, bool backoff, int threads) -> (Tensor[], Tensor, str)");
  m.def("classifier_saga_mv(Tensor numbers, Tensor identities, Tensor centers, "
        "Tensor scales, Tensor weights) -> Tensor");
  m.def("classifier_saga_prepare(Tensor rn, Tensor rc, Tensor sn, Tensor sc, "
        "Tensor rf, Tensor sf, int fold, int memory) -> Tensor[]");
  m.def("classifier_saga_preparation_rows(int n, bool enabled=True, int requested=4096) -> int");
  m.def("classifier_saga_preparation_sample(int n, int requested, int seed) -> Tensor");
}
TORCH_LIBRARY_IMPL(tabgen_eval_cpu, CPU, m) {
  m.impl("classifier_saga_updates", saga_updates);
  m.impl("classifier_saga_fit", saga_fit);
  m.impl("classifier_saga_guarded_updates", saga_guarded_updates);
  m.impl("classifier_saga_mv", saga_mv);
  m.impl("classifier_saga_prepare", saga_prepare);
}
TORCH_LIBRARY_IMPL(tabgen_eval_cpu, CatchAll, m) {
  m.impl("classifier_saga_preparation_rows", saga_preparation_rows);
  m.impl("classifier_saga_preparation_sample", saga_preparation_sample);
}
