/** @file
 * @brief Selected-precision Shape/Trend/Conditional reductions with bounded
 * categorical grids.
 */
#include "cpu/fidelity/fidelity.h"
#include "cpu/fidelity/statistics.h"
#include "cpu/runtime/thread_budget.h"
#include <type_traits>
namespace ltm {
template <class T> std::vector<T> correlation_matrix(const Sample<T>& s) {
  int64_t p = s.numeric.size();
  std::vector<T> rho(p * p, T(nan));
  if (!p || s.n < 2)
    return rho;
  bool complete = true;
  for (auto& c : s.numeric)
    for (auto x : c)
      complete &= std::isfinite(x);
  if (!complete) {
    for (int64_t i = 0; i < p; ++i)
      for (int64_t j = i + 1; j < p; ++j)
        rho[i * p + j] = rho[j * p + i] = pearson(s.numeric[i], s.numeric[j]);
    return rho;
  }
  auto type = std::is_same_v<T, double> ? at::kDouble : at::kFloat;
  auto x = at::empty({s.n, p}, at::TensorOptions().dtype(type));
  auto* xp = x.template data_ptr<T>();
  for (int64_t c = 0; c < p; ++c) {
    T anchor = s.numeric[c][0], scale = 0;
    for (auto v : s.numeric[c])
      scale = std::max(scale, std::abs(v - anchor));
    std::vector<T> z(s.n);
    if (std::isfinite(scale)) {
      for (int64_t i = 0; i < s.n; ++i)
        z[i] = scale > 0 ? (s.numeric[c][i] - anchor) / scale : T(0);
    } else {
      T absolute = 0;
      for (auto v : s.numeric[c])
        absolute = std::max(absolute, std::abs(v));
      for (int64_t i = 0; i < s.n; ++i)
        z[i] = s.numeric[c][i] / absolute - anchor / absolute;
    }
    T center = mean(z);
    for (int64_t i = 0; i < s.n; ++i)
      xp[i * p + c] = z[i] - center;
  }
  auto gram = at::mm(x.transpose(0, 1), x);
  const auto* g = gram.template const_data_ptr<T>();
  for (int64_t i = 0; i < p; ++i)
    for (int64_t j = i + 1; j < p; ++j)
      if (g[i * p + i] > 0 && g[j * p + j] > 0)
        rho[i * p + j] = rho[j * p + i] =
            std::clamp(g[i * p + j] / std::sqrt(g[i * p + i] * g[j * p + j]), T(-1), T(1));
  return rho;
}
template <class T>
std::vector<T> fidelity(const Sample<T>& a, const Sample<T>& b, const std::vector<int64_t>& levels,
                        bool global, int64_t bins, int threads, FidelityScoreDetail<T>* details,
                        int64_t components) {
  int64_t pn = a.numeric.size(), p = a.width();
  bool wants_shape = !global && (components & 1), wants_trend = components & 2;
  std::vector<T> shape;
  std::vector<std::vector<int64_t>> rc(p), sc(p);
  std::vector<int64_t> cards = levels;
  for (int64_t c = 0; c < pn; ++c) {
    auto r = (wants_shape || (wants_trend && p > pn)) ? sorted(a.numeric[c]) : std::vector<T>{};
    if (wants_shape)
      shape.push_back(ks(r, sorted(b.numeric[c])));
    if (wants_trend && p > pn) {
      auto e = edges_sorted(r, bins, true);
      cards.insert(cards.begin() + c, e.empty() ? 0 : e.size() - 1);
      rc[c] = bin(a.numeric[c], e, false, false);
      sc[c] = bin(b.numeric[c], e, true, global);
    }
  }
  if (p == pn)
    cards.insert(cards.begin(), pn, 0);
  for (int64_t c = pn; c < p; ++c) {
    int64_t j = c - pn;
    if (wants_trend) {
      rc[c] = a.categorical[j];
      sc[c] = b.categorical[j];
    }
    if (wants_shape)
      shape.push_back(tv<T>(a.categorical[j], b.categorical[j], levels[j]));
  }
  if (!wants_trend)
    return {mean(shape), T(nan)};
  auto ar = correlation_matrix(a), bs = correlation_matrix(b);
  std::vector<std::pair<int64_t, int64_t>> pairs;
  for (int64_t i = 0; i < p; ++i)
    for (int64_t j = i + 1; j < p; ++j)
      pairs.push_back({i, j});
  std::vector<T> scores(pairs.size(), T(nan));
#pragma omp parallel for num_threads(threads) if (threads > 1 && pairs.size() > 16)
  for (int64_t pair = 0; pair < (int64_t)pairs.size(); ++pair) {
    auto [i, j] = pairs[pair];
    if (j < pn) {
      T r = ar[i * pn + j], s = bs[i * pn + j];
      if (std::isfinite(r) && std::isfinite(s))
        scores[pair] = T(1) - T(.5) * std::abs(r - s);
    } else
      scores[pair] = contingency<T>(rc[i], rc[j], sc[i], sc[j], cards[i], cards[j]);
  }
  std::vector<T> result{mean(shape), mean(scores)};
  if (details) {
    details->pairs = std::move(pairs);
    details->scores = std::move(scores);
  }
  return result;
}
template <class T>
bool has_rare_sample(const Sample<T>& a, int64_t count, int64_t bins, double rare) {
  for (int64_t c = 0; c < a.width(); ++c) {
    auto labels = c < (int64_t)a.numeric.size()
                      ? bin(a.numeric[c], edges(a.numeric[c], bins, false), false, true)
                      : a.categorical[c - a.numeric.size()];
    std::map<int64_t, int64_t> n;
    int64_t total = 0;
    for (auto v : labels)
      if (v >= 0) {
        ++n[v];
        ++total;
      }
    for (auto kv : n)
      if (kv.second >= count && double(kv.second) / double(total) < rare)
        return true;
  }
  return false;
}
template <class T>
std::vector<T> conditional(const Sample<T>& a, const Sample<T>& b,
                           const std::vector<int64_t>& levels, int64_t count, int64_t bins,
                           double rare, int threads, ConditionalDetail<T>* details,
                           int64_t components) {
  int64_t p = a.width(), pn = a.numeric.size();
  bool wants_shape = (components & 1) && p >= 2;
  bool wants_trend = (components & 2) && p >= 3;
  bool wants_recovery = components & 4;
  std::vector<std::vector<T>> columns(p, std::vector<T>(6, T(nan)));
  if (details)
    details->slices.resize(p);
#pragma omp parallel for num_threads(threads) if (threads > 1 && p > 1)
  for (int64_t cond = 0; cond < p; ++cond) {
    ThreadBudget local(1);
    std::vector<int64_t> r, s;
    std::vector<T> condition_edges;
    if (cond < pn) {
      condition_edges = edges(a.numeric[cond], bins, false);
      if (condition_edges.empty())
        continue;
      r = bin(a.numeric[cond], condition_edges, false, true);
      s = bin(b.numeric[cond], condition_edges, true, true);
    } else {
      r = a.categorical[cond - pn];
      s = b.categorical[cond - pn];
    }
    int64_t nr = 0, ns = 0;
    std::map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> groups;
    for (int64_t i = 0; i < a.n; ++i)
      if (r[i] >= 0) {
        groups[r[i]].first.push_back(i);
        ++nr;
      }
    for (int64_t i = 0; i < b.n; ++i)
      if (s[i] >= 0) {
        ++ns;
        auto it = groups.find(s[i]);
        if (it != groups.end())
          it->second.second.push_back(i);
      }
    if (!nr)
      continue;
    std::vector<T> scores[6];
    std::vector<int64_t> cards = levels;
    if (cond >= pn)
      cards.erase(cards.begin() + cond - pn);
    for (auto& kv : groups) {
      auto& ri = kv.second.first;
      auto& si = kv.second.second;
      ConditionalSliceDetail<T>* detail = nullptr;
      if (details) {
        auto& rows = details->slices[cond];
        rows.emplace_back();
        detail = &rows.back();
        detail->label = kv.first;
        detail->reference_count = ri.size();
        detail->candidate_count = si.size();
        detail->reference_total = nr;
        detail->candidate_total = ns;
        detail->eligible = (int64_t)ri.size() >= count;
        detail->rare = double(ri.size()) / double(nr) < rare;
        if (cond < pn) {
          detail->lower = condition_edges[kv.first];
          detail->upper = condition_edges[kv.first + 1];
          detail->left_closed = kv.first == 0;
          detail->right_closed = true;
        }
      }
      if ((int64_t)ri.size() < count)
        continue;
      double frequency = double(ri.size()) / double(nr);
      // Rarity is a structural count decision; reported probabilities and
      // recovery aggregation use the selected numerical precision.
      T reference_frequency = T(ri.size()) / T(nr);
      T recovery =
          wants_recovery ? (ns ? (T(si.size()) / T(ns)) / reference_frequency : T(0)) : T(nan);
      T sh = wants_shape ? T(0) : T(nan), tr = wants_trend ? T(0) : T(nan);
      if (!si.empty() && (wants_shape || wants_trend)) {
        auto ar = select(a, ri, cond), bs = select(b, si, cond);
        auto result = fidelity(ar, bs, cards, false, bins, 1,
                               static_cast<FidelityScoreDetail<T>*>(nullptr), components & 3);
        sh = result[0];
        tr = result[1];
      }
      if (detail) {
        detail->shape = sh;
        detail->trend = tr;
        detail->recovery = recovery;
      }
      scores[0].push_back(sh);
      scores[2].push_back(tr);
      scores[4].push_back(recovery);
      if (frequency < rare) {
        scores[1].push_back(sh);
        scores[3].push_back(tr);
        scores[5].push_back(recovery);
      }
    }
    for (int i = 0; i < 6; ++i)
      columns[cond][i] = mean(scores[i]);
  }
  std::vector<T> result(6);
  for (int i = 0; i < 6; ++i) {
    std::vector<T> values;
    for (auto& column : columns)
      values.push_back(column[i]);
    result[i] = mean(values);
  }
  if (details)
    details->columns = std::move(columns);
  return result;
}
template std::vector<float> fidelity(const Sample<float>&, const Sample<float>&,
                                     const std::vector<int64_t>&, bool, int64_t, int,
                                     FidelityScoreDetail<float>*, int64_t);
template std::vector<double> fidelity(const Sample<double>&, const Sample<double>&,
                                      const std::vector<int64_t>&, bool, int64_t, int,
                                      FidelityScoreDetail<double>*, int64_t);
template std::vector<float> conditional(const Sample<float>&, const Sample<float>&,
                                        const std::vector<int64_t>&, int64_t, int64_t, double, int,
                                        ConditionalDetail<float>*, int64_t);
template std::vector<double> conditional(const Sample<double>&, const Sample<double>&,
                                         const std::vector<int64_t>&, int64_t, int64_t, double, int,
                                         ConditionalDetail<double>*, int64_t);
template bool has_rare_sample(const Sample<float>&, int64_t, int64_t, double);
template bool has_rare_sample(const Sample<double>&, int64_t, int64_t, double);
} // namespace ltm
