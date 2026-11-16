#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ltm::saga::calibration {
// Gazagnadou et al. (2019), Eq. 20, is a practical smoothness estimate.
// The power and sampled-maximum margins below are empirical, not certificates.
struct Rate {
  double data_raw, data_used, maximum_raw, maximum_used, ridge_upper;
  double practical, rate;
  int64_t rows, population, batch;
};
inline Rate rate(int64_t n, int64_t requested_batch, int64_t rows, double data_raw,
                 double maximum_raw, double ridge_upper) {
  if (n < 1 || requested_batch < 1 || rows < 1 || rows > n || !std::isfinite(data_raw) ||
      data_raw <= 0 || !std::isfinite(maximum_raw) || maximum_raw <= 0 ||
      !std::isfinite(ridge_upper) || ridge_upper < 0)
    throw std::invalid_argument("invalid SketchySAGA rate calibration");
  const int64_t b = std::min(requested_batch, n);
  const double data = 2 * data_raw, maximum = std::max(2 * maximum_raw, data);
  const double a = n == 1 ? 1. : double(n - b) / (double(b) * (n - 1));
  const double c = n == 1 ? 0. : double(n) * (b - 1) / (double(b) * (n - 1));
  const double practical = c * data + a * maximum;
  const double eta = 1 / (practical + ridge_upper);
  if (!std::isfinite(eta) || eta <= 0)
    throw std::invalid_argument("nonfinite SketchySAGA calibrated rate");
  return {data_raw, data, maximum_raw, maximum, ridge_upper, practical, eta, rows, n, b};
}
} // namespace ltm::saga::calibration
