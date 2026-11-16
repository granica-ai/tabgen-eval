/** @file Single-point fidelity execution and typed descriptive output packing. */
#include "cpu/fidelity/fidelity.h"
#include "cpu/fidelity/shape.h"
#include "cpu/runtime/draw_tasks.h"
#include "cpu/runtime/tensor_validation.h"
#include "cpu/runtime/thread_budget.h"
#include <array>
#include <numeric>
#include <type_traits>

namespace ltm {
namespace {
using DetailOutput = c10::Dict<std::string, Tensor>;

template <class T>
void put(DetailOutput& out, const std::string& name, const std::vector<T>& values) {
  auto kind = std::is_same_v<T, int64_t> ? at::kLong
              : std::is_same_v<T, float> ? at::kFloat
                                         : at::kDouble;
  out.insert(name, at::tensor(values, at::TensorOptions().dtype(kind)));
}

void flags(DetailOutput& out, const std::string& name, const std::vector<int64_t>& values) {
  out.insert(name, at::tensor(values, at::TensorOptions().dtype(at::kLong)).to(at::kBool));
}

void indices(DetailOutput& out, int64_t width) {
  std::vector<int64_t> positions(width);
  std::iota(positions.begin(), positions.end(), 0);
  put(out, "columns.column_index", positions);
}

template <class T> void pack_conditional(DetailOutput& out, const ConditionalDetail<T>& detail) {
  static const std::array<const char*, 6> names = {"cond_shape_all", "cond_shape_rare",
                                                   "cond_trend_all", "cond_trend_rare",
                                                   "recovery_all",   "recovery_rare"};
  indices(out, detail.columns.size());
  for (int stat = 0; stat < 6; ++stat) {
    std::vector<T> values;
    values.reserve(detail.columns.size());
    for (const auto& column : detail.columns)
      values.push_back((detail.columns.size() < 2 && stat < 2) ||
                               (detail.columns.size() < 3 && (stat == 2 || stat == 3))
                           ? T(nan)
                           : column[stat]);
    put(out, std::string("columns.") + names[stat], values);
  }
  std::vector<int64_t> column, label, nr, ns, total_r, total_s;
  std::vector<int64_t> eligible, rare, left_closed, right_closed;
  std::vector<T> shape, trend, recovery, lower, upper;
  for (size_t c = 0; c < detail.slices.size(); ++c)
    for (const auto& row : detail.slices[c]) {
      column.push_back(c);
      label.push_back(row.label);
      nr.push_back(row.reference_count);
      ns.push_back(row.candidate_count);
      total_r.push_back(row.reference_total);
      total_s.push_back(row.candidate_total);
      eligible.push_back(row.eligible);
      rare.push_back(row.rare);
      shape.push_back(detail.columns.size() < 2 ? T(nan) : row.shape);
      trend.push_back(detail.columns.size() < 3 ? T(nan) : row.trend);
      recovery.push_back(row.recovery);
      lower.push_back(row.lower);
      upper.push_back(row.upper);
      left_closed.push_back(row.left_closed);
      right_closed.push_back(row.right_closed);
    }
  put(out, "slices.column_index", column);
  put(out, "slices.label_code", label);
  put(out, "slices.reference_count", nr);
  put(out, "slices.candidate_count", ns);
  put(out, "slices.reference_total", total_r);
  put(out, "slices.candidate_total", total_s);
  flags(out, "slices.eligible", eligible);
  flags(out, "slices.rare", rare);
  put(out, "slices.shape", shape);
  put(out, "slices.trend", trend);
  put(out, "slices.recovery", recovery);
  put(out, "slices.lower", lower);
  put(out, "slices.upper", upper);
  flags(out, "slices.left_closed", left_closed);
  flags(out, "slices.right_closed", right_closed);
}

template <class T>
DetailOutput detailed_t(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
                        const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
                        const std::vector<int64_t>& cards, const std::string& metric,
                        int64_t minimum, int64_t bins, double rare, int threads,
                        int64_t components) {
  DetailOutput out;
  std::vector<T> result;
  ThreadBudget budget(threads);
  if (metric == "shape") {
    std::vector<T> columns;
    result = {shape<T>(rn, rc, sn, sc, rd.const_data_ptr<int64_t>(), rd.size(1),
                       sd.const_data_ptr<int64_t>(), sd.size(1), cards, threads, &columns)};
    indices(out, columns.size());
    put(out, "columns.score", columns);
  } else {
    auto a = gather<T>(rn, rc, rm, rd.const_data_ptr<int64_t>(), rd.size(1));
    auto b = gather<T>(sn, sc, sm, sd.const_data_ptr<int64_t>(), sd.size(1));
    if (metric == "trend") {
      FidelityScoreDetail<T> detail;
      result = {fidelity(a, b, cards, true, bins, threads, &detail)[1]};
      std::vector<int64_t> first, second;
      first.reserve(detail.pairs.size());
      second.reserve(detail.pairs.size());
      for (auto pair : detail.pairs) {
        first.push_back(pair.first);
        second.push_back(pair.second);
      }
      put(out, "pairs.column_1_index", first);
      put(out, "pairs.column_2_index", second);
      put(out, "pairs.score", detail.scores);
    } else {
      ConditionalDetail<T> detail;
      result = conditional(a, b, cards, minimum, bins, rare, threads, &detail, components);
      pack_conditional(out, detail);
    }
  }
  put(out, "values", std::vector<double>(result.begin(), result.end()));
  return out;
}
} // namespace

c10::Dict<std::string, Tensor>
fidelity_details(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
                 const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
                 std::vector<int64_t> cards, std::string metric, int64_t minimum, int64_t bins,
                 double rare, int64_t threads, int64_t components) {
  validate_pair(rn, rc, sn, sc, rm, sm, rd, sd, cards);
  TORCH_CHECK(rd.size(0) == 1, "fidelity details require exactly one draw");
  TORCH_CHECK(metric == "shape" || metric == "trend" || metric == "conditional_fidelity",
              "unsupported detailed fidelity metric");
  TORCH_CHECK(threads > 0 && threads <= INT32_MAX && minimum > 0 && bins > 0 &&
                  std::isfinite(rare) && rare >= 0 && rare <= 1 && components >= 0 &&
                  components <= 7,
              "invalid CPU fidelity detail parameters");
  validate_category_draws(rc, rd, cards);
  validate_category_draws(sc, sd, cards);
  return rn.scalar_type() == at::kDouble
             ? detailed_t<double>(rn, rc, sn, sc, rm, sm, rd, sd, cards, metric, minimum, bins,
                                  rare, threads, components)
             : detailed_t<float>(rn, rc, sn, sc, rm, sm, rd, sd, cards, metric, minimum, bins, rare,
                                 threads, components);
}
} // namespace ltm
