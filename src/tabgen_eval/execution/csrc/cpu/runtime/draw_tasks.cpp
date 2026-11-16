#include "cpu/runtime/draw_tasks.h"
#include "cpu/distance/distance.h"
#include "cpu/fidelity/fidelity.h"
#include "cpu/fidelity/shape.h"
#include "cpu/runtime/tensor_validation.h"
#include "cpu/runtime/thread_budget.h"

namespace ltm {
void validate_pair(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
                   const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
                   const std::vector<int64_t>& cards) {
  validate_pool(rn, rc, rm);
  validate_pool(sn, sc, sm);
  validate_draws(rd, rn.size(0));
  validate_draws(sd, sn.size(0));
  TORCH_CHECK(rn.scalar_type() == sn.scalar_type() && rn.size(1) == sn.size(1) &&
                  rc.size(1) == sc.size(1) && rc.size(1) == (int64_t)cards.size() &&
                  rd.size(0) == sd.size(0),
              "incompatible pools/draw batches");
  for (auto c : cards)
    TORCH_CHECK(c >= 0, "negative cardinality");
}

template <class T>
Tensor samples_t(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
                 const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
                 const std::vector<int64_t>& cards, const std::string& metric, int64_t k,
                 int64_t minimum, int64_t bins, double rare, int64_t threads, int64_t components) {
  int64_t width = metric == "prdc" ? 4 : metric == "conditional_fidelity" ? 6 : 1;
  auto out = at::empty({rd.size(0), width}, rn.options().dtype(at::kDouble));
  auto* p = out.data_ptr<double>();
  int outer = rd.size(0) > 1 ? std::min<int64_t>(threads, rd.size(0)) : 1;
#pragma omp parallel for num_threads(outer) if (outer > 1)
  for (int64_t task = 0; task < rd.size(0); ++task) {
    int inner = outer > 1 ? 1 : threads;
    ThreadBudget budget(inner);
    if (metric == "shape") {
      p[task] = double(shape<T>(rn, rc, sn, sc, rd.const_data_ptr<int64_t>() + task * rd.size(1),
                                rd.size(1), sd.const_data_ptr<int64_t>() + task * sd.size(1),
                                sd.size(1), cards, inner));
      continue;
    }
    auto a = gather<T>(rn, rc, rm, rd.const_data_ptr<int64_t>() + task * rd.size(1), rd.size(1));
    auto b = gather<T>(sn, sc, sm, sd.const_data_ptr<int64_t>() + task * sd.size(1), sd.size(1));
    std::vector<double> result;
    if (metric == "prdc")
      result = prdc(a, b, k, inner, components);
    else if (metric == "trend")
      result = {double(fidelity(a, b, cards, true, bins, inner)[1])};
    else {
      auto values = conditional(a, b, cards, minimum, bins, rare, inner,
                                static_cast<ConditionalDetail<T>*>(nullptr), components);
      result.assign(values.begin(), values.end());
    }
    for (int64_t i = 0; i < width; ++i)
      p[task * width + i] = result[i];
  }
  return out;
}
/** @brief Execute complete CPU Shape/Trend/Conditional/PRDC draw tasks. Numeric
 * sources preserve selected float/double; int64 [B,Ndraw] positions preserve
 * duplicates. Returns CPU double [B,K] report scores; malformed metadata raises
 * before parallel work and sources remain read-only. */
Tensor samples(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
               const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
               std::vector<int64_t> cards, std::string metric, int64_t k, int64_t minimum,
               int64_t bins, double rare, int64_t threads, int64_t components) {
  validate_pair(rn, rc, sn, sc, rm, sm, rd, sd, cards);
  TORCH_CHECK(threads > 0 && threads <= INT32_MAX && k > 0 && minimum > 0 && bins > 0 &&
                  std::isfinite(rare) && rare >= 0 && rare <= 1 && components >= 0 &&
                  components <= 15,
              "invalid CPU metric parameters");
  TORCH_CHECK(metric == "shape" || metric == "trend" || metric == "prdc" ||
                  metric == "conditional_fidelity",
              "unsupported CPU sample metric");
  // Validate only rows consumed by this batch, in their contiguous row layout.
  // A small evaluation draw must not scan every cell of a large source pool.
  validate_category_draws(rc, rd, cards);
  validate_category_draws(sc, sd, cards);
  return rn.scalar_type() == at::kDouble
             ? samples_t<double>(rn, rc, sn, sc, rm, sm, rd, sd, cards, metric, k, minimum, bins,
                                 rare, threads, components)
             : samples_t<float>(rn, rc, sn, sc, rm, sm, rd, sd, cards, metric, k, minimum, bins,
                                rare, threads, components);
}
} // namespace ltm
