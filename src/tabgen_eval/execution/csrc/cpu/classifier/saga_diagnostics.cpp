#include "common/classifier/saga_preparation.hpp"
#include "cpu/classifier/detail/saga.h"

namespace ltm::saga {
using detail::chunk_epochs;
using detail::probe_limit;
using detail::retry_limit;
long double guard_bytes(const Design& x) {
  // Fixed probe, gathered logits/loss temporaries, sampling scratch and one
  // reusable complete compact-state snapshot. No dense one-hot expansion.
  const auto m = std::min(probe_limit, x.n());
  return x.numbers.element_size() * (m * (x.pn() + 9) + x.n() + 4 * x.d()) + 8.L * m * x.pc();
}
std::vector<double> guard_diagnostics(const Fit& r) {
  return {double(r.accepted_epochs),
          double(r.accepted_updates),
          r.initial_rate,
          r.accepted_rate,
          r.attempt_rate,
          double(r.backoff_enabled),
          double(r.probe_rows),
          double(r.probe_checks),
          double(r.backoffs),
          double(r.rejected_epochs),
          double(retry_limit),
          r.probe_objective,
          double(chunk_epochs)};
}
std::vector<double> preparation_diagnostics(const Fit& r) {
  std::vector<double> out = {double(r.preparation_training_rows), double(r.preparation_rows),
                             double(r.sampled_preparation), double(r.preparation_sample_hash)};
  out.insert(out.end(), r.rate_calibration.begin(), r.rate_calibration.end());
  return out;
}
long double workspace_bytes(const Design& x, int64_t batch, int64_t rank, bool backoff,
                            bool sampled, int64_t requested_rows) {
  int64_t b = std::min(batch, x.n()), r = std::min(rank, x.d());
  int64_t m = ltm_saga_preparation::rows(x.n(), sampled, requested_rows);
  int64_t calibration_n = std::min<int64_t>(4096, m);
  long double bytes =
      x.numbers.element_size() * (x.n() * x.pn() + x.k() * 2 + 2 * x.n() + 16.L * x.d() * r +
                                  16 * x.d() + b * (x.pn() + 8)) +
      8.L * (x.n() + b) * x.pc();
  if (m < x.n())
    bytes += x.numbers.element_size() * m * (x.pn() + 1) + 8.L * m * x.pc();
  // Calibration gather, squared numeric scratch, skinny projections and
  // transformed basis. The full n-row gradient cache remains unchanged.
  bytes += x.numbers.element_size() *
               (calibration_n * (2 * x.pn() + 2 * r + 8) + 4.L * x.d() * r + 8 * x.d()) +
           8.L * calibration_n * x.pc();
  return bytes + (backoff ? guard_bytes(x) : 0);
}

Tensor fit_diagnostics(const Fit& r, const at::TensorOptions& options) {
  std::vector<double> values = {double(r.converged), double(r.iterations),
                                double(r.updates),   r.gradient_squared,
                                r.relative_change,   r.rate,
                                double(r.rank),      double(r.stable_inverse),
                                r.gradient_infinity, r.initial_gradient_infinity,
                                r.gradient_threshold};
  auto guard = ltm::saga::guard_diagnostics(r);
  values.insert(values.end(), guard.begin(), guard.end());
  auto preparation = ltm::saga::preparation_diagnostics(r);
  values.insert(values.end(), preparation.begin(), preparation.end());
  // Diagnostic scalars and integer seed hashes retain exact FP64 representation.
  return at::tensor(values, options.dtype(at::kDouble));
}

} // namespace ltm::saga
