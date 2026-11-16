#include "cpu/classifier/detail/newton.h"
#include <algorithm>
#include <cmath>

namespace ltm::classifier_detail {
Tensor expand(const saga::Design& x) {
  return AT_DISPATCH_FLOATING_TYPES(x.numbers.scalar_type(), "cpu_classifier", [&]() {
    auto result = at::empty({x.d(), x.n()}, x.numbers.options()).t();
    if (x.pn())
      result.narrow(1, 0, x.pn()).copy_(x.numbers);
    auto* out = result.data_ptr<scalar_t>();
    const auto* centers = x.centers.const_data_ptr<scalar_t>();
    const auto* scales = x.scales.const_data_ptr<scalar_t>();
    for (int64_t j = 0; j < x.k(); ++j)
      for (int64_t r = 0; r < x.n(); ++r)
        out[(x.pn() + j) * x.n() + r] = -centers[j] / scales[j];
    const auto* ids = x.identities.const_data_ptr<int64_t>();
    for (int64_t r = 0; r < x.n(); ++r)
      for (int64_t c = 0; c < x.pc(); ++c) {
        auto id = ids[r * x.pc() + c];
        if (id >= 0)
          out[(x.pn() + id) * x.n() + r] += 1 / scales[id];
      }
    result.select(1, x.d() - 1).fill_(1);
    return result;
  });
}
double auc(const Tensor& scores, const Tensor& labels) {
  return AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "cpu_classifier", [&]() {
    const auto* z = scores.const_data_ptr<scalar_t>();
    const auto* y = labels.const_data_ptr<scalar_t>();
    std::vector<std::pair<double, bool>> order;
    int64_t positives = 0;
    for (int64_t i = 0; i < scores.numel(); ++i) {
      if (!std::isfinite(z[i]))
        return nan;
      order.push_back({z[i], y[i] == 1});
      positives += y[i] == 1;
    }
    if (!positives || positives == scores.numel())
      return nan;
    std::sort(order.begin(), order.end(), [](auto a, auto b) { return a.first < b.first; });
    int64_t below = 0;
    double wins = 0;
    for (size_t start = 0; start < order.size();) {
      size_t end = start + 1;
      while (end < order.size() && order[end].first == order[start].first)
        ++end;
      int64_t p = 0;
      for (size_t i = start; i < end; ++i)
        p += order[i].second;
      int64_t n = end - start - p;
      wins += double(p) * (double(below) + .5 * n);
      below += n;
      start = end;
    }
    return wins / (double(positives) * (scores.numel() - positives));
  });
}

} // namespace ltm::classifier_detail
