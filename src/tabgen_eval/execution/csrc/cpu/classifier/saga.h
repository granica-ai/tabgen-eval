#pragma once
#include "cpu/classifier/diagnostics.h"
#include "cpu/runtime/sample.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ltm::saga {
struct NumericalFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// Compact full-one-hot design: identities contain categorical feature offsets,
// -1 for unseen levels. The final coefficient is an unpenalized intercept.
struct Design {
  Tensor numbers, identities, centers, scales;
  int64_t n() const {
    return numbers.size(0);
  }
  int64_t pn() const {
    return numbers.size(1);
  }
  int64_t pc() const {
    return identities.size(1);
  }
  int64_t k() const {
    return centers.numel();
  }
  int64_t d() const {
    return pn() + k() + 1;
  }
};
struct Fold {
  Design train, test;
  Tensor labels, scoring;
  std::vector<Tensor> preprocessing;
};
struct Fit {
  Tensor coefficients;
  bool converged = false;
  int64_t iterations = 0, updates = 0, rank = 0;
  double gradient_squared = nan, relative_change = nan, rate = nan;
  double maximum_coefficient = nan, maximum_change = nan, intercept_change = nan, eigenvalue = nan;
  double gradient_infinity = nan, initial_gradient_infinity = nan, gradient_threshold = nan;
  std::string reason = "max_epochs";
  bool stable_inverse = false;
  bool backoff_enabled = false;
  int64_t accepted_epochs = 0, accepted_updates = 0, probe_rows = 0, probe_checks = 0, backoffs = 0,
          rejected_epochs = 0;
  double initial_rate = nan, accepted_rate = nan, attempt_rate = nan, probe_objective = nan;
  int64_t preparation_training_rows = 0, preparation_rows = 0, preparation_sample_hash = 0;
  bool sampled_preparation = false;
  std::vector<double> rate_calibration = std::vector<double>(9, nan);
};
// Appended ABI fields, shared by the standalone and batched CPU fit operators.
constexpr int64_t guard_diagnostic_count = classifier_diagnostics::guard::width;
std::vector<double> guard_diagnostics(const Fit& fit);
Tensor fit_diagnostics(const Fit& fit, const at::TensorOptions& options);
constexpr int64_t preparation_diagnostic_count = classifier_diagnostics::preparation::width;
std::vector<double> preparation_diagnostics(const Fit& fit);
long double guard_bytes(const Design& design);
long double workspace_bytes(const Design& design, int64_t batch, int64_t rank, bool backoff,
                            bool sampled, int64_t preparation_rows);
template <class T>
std::optional<Fold> prepare(const Sample<T>& a, const Sample<T>& b, const int64_t* af,
                            const int64_t* bf, int64_t fold, int64_t memory);
Tensor mv(const Design& design, const Tensor& weights);
Fit fit(const Design& design, const Tensor& labels, double ridge, double tolerance, int64_t epochs,
        int64_t batch_size, int64_t rank, uint64_t seed, int64_t memory, bool backoff = true,
        bool sampled = true, int64_t preparation_rows = 4096);
} // namespace ltm::saga
