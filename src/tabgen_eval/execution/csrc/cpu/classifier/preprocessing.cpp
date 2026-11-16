#include "common/preprocessing/categories.hpp"
#include "common/preprocessing/numeric.hpp"
#include "cpu/classifier/saga.h"
#include <cmath>
#include <unordered_map>

namespace ltm::saga {
// Compact solver storage uses the same training-only policy primitives as the
// standalone preprocessing operators; categorical indicators remain implicit.
template <class T>
std::optional<Fold> prepare(const Sample<T>& a, const Sample<T>& b, const int64_t* af,
                            const int64_t* bf, int64_t fold, int64_t memory) {
  std::vector<int64_t> rows[4];
  for (int64_t i = 0; i < a.n; ++i)
    rows[af[i] == fold ? 2 : 0].push_back(i);
  for (int64_t i = 0; i < b.n; ++i)
    rows[bf[i] == fold ? 3 : 1].push_back(i);
  for (const auto& group : rows)
    if (group.empty())
      return std::nullopt;
  int64_t nt = rows[0].size() + rows[1].size(), nv = rows[2].size() + rows[3].size();
  std::vector<std::pair<int64_t, preprocessing::NumericColumn>> numeric;
  std::vector<int64_t> indicators;
  for (int64_t c = 0; c < int64_t(a.numeric.size()); ++c) {
    auto state = preprocessing::fit_numeric_column(
        rows[0].size(), rows[1].size(),
        [&](int g, int64_t r) { return (g ? b : a).numeric[c][rows[g][r]]; }, true);
    if (state.keep) {
      numeric.emplace_back(c, state);
      if (state.indicator)
        indicators.push_back(c);
    }
  }
  std::vector<std::vector<int64_t>> vocab;
  int64_t k = 0;
  for (int64_t c = 0; c < int64_t(a.categorical.size()); ++c) {
    vocab.push_back(
        preprocessing::active_categories(rows[0].size(), rows[1].size(), [&](int g, int64_t r) {
          return (g ? b : a).categorical[c][rows[g][r]];
        }));
    k += vocab.back().size();
  }
  int64_t pn = numeric.size() + indicators.size(), pc = vocab.size();
  long double bytes = sizeof(T) * ((nt + nv) * (pn + 1) + 2.L * k + nt * 4 + 32 * (pn + k + 1)) +
                      8.L * (nt + nv) * pc + 24.L * k +
                      8.L * (numeric.size() + indicators.size() + 4 * a.numeric.size() + 6 * pc);
  if (bytes > memory)
    return std::nullopt;
  const at::TensorOptions opts =
      at::TensorOptions().dtype(std::is_same_v<T, double> ? at::kDouble : at::kFloat);
  auto ids = opts.dtype(at::kLong);
  Fold out{{at::empty({nt, pn}, opts), at::empty({nt, pc}, ids), at::zeros({k}, opts),
            at::ones({k}, opts)},
           {at::empty({nv, pn}, opts), at::empty({nv, pc}, ids), {}, {}},
           at::zeros({nt}, opts),
           at::zeros({nv}, opts)};
  out.test.centers = out.train.centers;
  out.test.scales = out.train.scales;
  out.labels.narrow(0, rows[0].size(), rows[1].size()).fill_(1);
  out.scoring.narrow(0, rows[2].size(), rows[3].size()).fill_(1);
  // Compact provenance is emitted by the same passes that apply the fit.
  auto keep = at::empty({int64_t(numeric.size())}, ids);
  auto flags = at::empty({int64_t(indicators.size())}, ids);
  auto active = at::empty({k, 3}, ids);
  auto numeric_missing = at::zeros({int64_t(a.numeric.size()), 4}, ids);
  auto category_missing = at::zeros({pc, 4}, ids);
  auto unseen = at::zeros({pc, 2}, ids);
  std::vector<int64_t> positions(a.numeric.size(), -1), flag_positions(a.numeric.size(), -1);
  for (size_t c = 0; c < numeric.size(); ++c) {
    keep.data_ptr<int64_t>()[c] = numeric[c].first;
    positions[numeric[c].first] = c;
  }
  for (size_t c = 0; c < indicators.size(); ++c) {
    flags.data_ptr<int64_t>()[c] = indicators[c];
    flag_positions[indicators[c]] = numeric.size() + c;
  }
  out.preprocessing = {keep, flags, active, numeric_missing, category_missing, unseen};
  bool finite = true;
  for (int g = 0; g < 4; ++g) {
    auto* dst = (g < 2 ? out.train.numbers : out.test.numbers).data_ptr<T>();
    const auto& source = g % 2 ? b : a;
    int64_t offset = g % 2 ? rows[g - 1].size() : 0;
    for (int64_t r = 0; r < int64_t(rows[g].size()); ++r)
      for (size_t c = 0; c < source.numeric.size(); ++c) {
        T value = source.numeric[c][rows[g][r]];
        bool missing = !std::isfinite(value);
        numeric_missing.data_ptr<int64_t>()[c * 4 + g] += missing;
        int64_t position = positions[c];
        if (position >= 0) {
          const auto& state = numeric[position].second;
          T z = ((missing ? T(state.impute) : value) - T(state.center)) / T(state.scale);
          dst[(offset + r) * pn + position] = z;
          finite &= std::isfinite(z);
        }
        if (flag_positions[c] >= 0)
          dst[(offset + r) * pn + flag_positions[c]] = missing;
      }
  }
  int64_t offset = 0;
  for (int64_t c = 0; c < pc; ++c) {
    std::unordered_map<int64_t, int64_t> map;
    for (int64_t j = 0; j < int64_t(vocab[c].size()); ++j) {
      map[vocab[c][j]] = offset + j;
      auto* record = active.data_ptr<int64_t>() + (offset + j) * 3;
      record[0] = c;
      record[1] = vocab[c][j];
      record[2] = offset + j;
    }
    for (int g = 0; g < 4; ++g) {
      auto* dst = (g < 2 ? out.train.identities : out.test.identities).data_ptr<int64_t>();
      int64_t rowoffset = g % 2 ? rows[g - 1].size() : 0;
      for (int64_t r = 0; r < int64_t(rows[g].size()); ++r) {
        auto code = (g % 2 ? b : a).categorical[c][rows[g][r]];
        auto found = map.find(code);
        dst[(rowoffset + r) * pc + c] = found == map.end() ? -1 : found->second;
        category_missing.data_ptr<int64_t>()[c * 4 + g] += code == -1;
        if (g >= 2 && code >= 0)
          unseen.data_ptr<int64_t>()[c * 2 + g - 2] += found == map.end();
      }
    }
    offset += vocab[c].size();
  }
  return finite ? std::optional<Fold>(std::move(out)) : std::nullopt;
}

template std::optional<Fold> prepare(const Sample<float>&, const Sample<float>&, const int64_t*,
                                     const int64_t*, int64_t, int64_t);
template std::optional<Fold> prepare(const Sample<double>&, const Sample<double>&, const int64_t*,
                                     const int64_t*, int64_t, int64_t);
} // namespace ltm::saga
