#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ltm::preprocessing {
// Stable FP64 accumulators over selected-precision observations; no wide data copy.
struct Moments {
  int64_t count = 0;
  double mean = 0, squares = 0;
  void add(double value) {
    ++count;
    double delta = value - mean;
    mean += delta / count;
    squares += delta * (value - mean);
  }
};
struct NumericColumn {
  bool keep = false, indicator = false;
  double impute = 0, center = 0, scale = 1;
};
// Getter accepts (class, row). Fit decisions consume training observations only.
template <class Getter>
NumericColumn fit_numeric_column(int64_t nr, int64_t ns, Getter value, bool linear) {
  Moments finite[2];
  bool missing = false;
  for (int g = 0; g < 2; ++g)
    for (int64_t r = 0; r < (g ? ns : nr); ++r) {
      double v = value(g, r);
      if (std::isfinite(v))
        finite[g].add(v);
      else
        missing = true;
    }
  NumericColumn out;
  out.keep = finite[0].count || finite[1].count;
  if (!out.keep)
    return out;
  out.indicator = linear && missing;
  out.impute = finite[0].count ? finite[0].mean : finite[1].mean;
  if (!linear)
    return out;
  Moments combined;
  for (int g = 0; g < 2; ++g)
    for (int64_t r = 0; r < (g ? ns : nr); ++r) {
      double v = value(g, r);
      combined.add(std::isfinite(v) ? v : out.impute);
    }
  out.center = combined.mean;
  double variance = std::max(0., combined.squares / combined.count);
  double eps = std::numeric_limits<double>::epsilon();
  double bound = combined.count * eps * variance + std::pow(combined.count * out.center * eps, 2);
  out.scale = variance <= bound ? 1 : std::sqrt(variance);
  return out;
}
} // namespace ltm::preprocessing
