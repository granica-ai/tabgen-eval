#include "cpu/classifier/detail/saga.h"
#include <optional>
#include <unordered_set>

namespace ltm::saga::detail {
namespace {
void update(const Design& x, const Tensor& labels, const Tensor& rows, State& state, double ridge,
            double rate, const Inverse* inverse) {
  Design batch{x.numbers.index_select(0, rows), x.identities.index_select(0, rows), x.centers,
               x.scales};
  auto logits = mv(batch, state.weights);
  auto residual = at::sigmoid(logits) - labels.index_select(0, rows);
  // Stable tails avoid 1-sigmoid cancellation for positive examples.
  residual = at::where(labels.index_select(0, rows) == 1, -at::sigmoid(-logits), residual);
  auto delta = residual - state.table.index_select(0, rows);
  auto correction = adjoint(batch, delta / rows.numel());
  state.gradient = state.average + correction;
  state.gradient.narrow(0, 0, x.d() - 1).add_(state.weights.narrow(0, 0, x.d() - 1), ridge);
  state.average.add_(correction, double(rows.numel()) / x.n());
  state.table.index_copy_(0, rows, residual);
  state.weights.add_(inverse ? inverse->apply(state.gradient) : state.gradient, -rate);
}
std::pair<double, double> change(State& state) {
  auto w = state.weights.narrow(0, 0, state.weights.numel() - 1),
       previous = state.previous.narrow(0, 0, state.weights.numel() - 1);
  double magnitude = w.abs().max().item<double>();
  double difference = (w - previous).abs().max().item<double>();
  state.previous.copy_(state.weights);
  return {magnitude, difference};
}

void sample_rows(int64_t n, Tensor& rows, std::mt19937_64& rng,
                 std::unordered_set<int64_t>& chosen) {
  chosen.clear();
  auto* row = rows.data_ptr<int64_t>();
  int64_t t = 0;
  // Floyd sampling: O(batch) memory, private RNG, uniform without replacement.
  for (int64_t j = n - rows.numel(); j < n; ++j) {
    std::uniform_int_distribution<int64_t> pick(0, j);
    auto k = pick(rng);
    auto value = chosen.count(k) ? j : k;
    chosen.insert(value);
    row[t++] = value;
  }
}

struct ObjectiveGuard {
  Design probe;
  Tensor labels;
  double ridge, previous = std::log(2.);
  ObjectiveGuard(const Design& x, const Tensor& y, double penalty, uint64_t seed) : ridge(penalty) {
    std::mt19937_64 rng(seed ^ 0xC19E);
    auto rows = at::empty({std::min(probe_limit, x.n())}, x.identities.options());
    std::unordered_set<int64_t> chosen;
    chosen.reserve(rows.numel() * 2);
    sample_rows(x.n(), rows, rng, chosen);
    probe = {x.numbers.index_select(0, rows), x.identities.index_select(0, rows), x.centers,
             x.scales};
    labels = y.index_select(0, rows);
  }
  double objective(const Tensor& weights) const {
    auto logits = mv(probe, weights);
    // Stable logistic loss, including extreme signed logits. Ridge applies
    // to feature coefficients only; the final intercept is unpenalized.
    auto loss = at::clamp_min(logits, 0) + at::log1p(at::exp(-at::abs(logits))) - labels * logits;
    auto features = weights.narrow(0, 0, weights.numel() - 1);
    return loss.mean(at::kDouble).item<double>() +
           .5 * ridge * at::dot(features, features).item<double>();
  }
  bool rejects(double value, bool finite) const {
    return !finite || !std::isfinite(value) ||
           value > previous + std::max(1e-4, .02 * std::abs(previous));
  }
};

State clone_state(const State& state) {
  return {state.weights.clone(), state.table.clone(), state.average.clone(), state.gradient.clone(),
          state.previous.clone()};
}
void copy_state(State& target, const State& source) {
  target.weights.copy_(source.weights);
  target.table.copy_(source.table);
  target.average.copy_(source.average);
  target.gradient.copy_(source.gradient);
  target.previous.copy_(source.previous);
}

// Shared by production fitting and prescribed-factor/schedule numerical
// oracles. Chunk rollback stays native, including state and sampler replay.
} // namespace

Fit run(const Design& x, const Tensor& labels, double ridge, double tolerance, int64_t epochs,
        int64_t batch, uint64_t seed, double rate, const Inverse* inverse, bool backoff,
        State& state, const Tensor& schedule) {
  Fit result;
  result.rate = result.initial_rate = rate;
  result.backoff_enabled = backoff;
  std::optional<ObjectiveGuard> guard;
  if (backoff) {
    guard.emplace(x, labels, ridge, seed);
    result.probe_rows = guard->probe.n();
    result.probe_objective = guard->previous;
  }
  std::optional<State> checkpoint;
  if (backoff)
    checkpoint = clone_state(state);
  std::mt19937_64 sample_rng(seed ^ 0xA617);
  std::unordered_set<int64_t> chosen;
  chosen.reserve(batch * 2);
  auto rows = at::empty({batch}, x.identities.options());
  int64_t steps = (x.n() + batch - 1) / batch;
  bool terminal_numerical = false;
  while (result.iterations < epochs) {
    int64_t count = std::min(chunk_epochs, epochs - result.iterations);
    auto sampler_checkpoint = sample_rng;
    int64_t schedule_checkpoint = result.accepted_epochs;
    if (checkpoint)
      copy_state(*checkpoint, state);
    bool accepted = false;
    while (result.iterations < epochs) {
      int64_t allowed = std::min(count, epochs - result.iterations), done = 0;
      Fit proposed;
      bool finite = true;
      for (int64_t e = 0; e < allowed; ++e) {
        for (int64_t s = 0; s < steps; ++s) {
          if (schedule.defined())
            rows = schedule.select(0, schedule_checkpoint + e).select(0, s);
          else
            sample_rows(x.n(), rows, sample_rng, chosen);
          update(x, labels, rows, state, ridge, rate, inverse);
          if (result.updates == 0)
            result.initial_gradient_infinity = state.gradient.abs().max().item<double>();
          ++result.updates;
        }
        ++done;
        ++result.iterations;
        proposed.intercept_change = std::abs(state.weights.select(0, x.d() - 1).item<double>() -
                                             state.previous.select(0, x.d() - 1).item<double>());
        auto [magnitude, difference] = change(state);
        proposed.maximum_coefficient = magnitude;
        proposed.maximum_change = difference;
        proposed.relative_change = magnitude ? difference / magnitude : (difference ? INFINITY : 0);
        proposed.gradient_infinity = state.gradient.abs().max().item<double>();
        proposed.gradient_threshold = tolerance;
        bool finite_weights = at::isfinite(state.weights).all().item<bool>();
        bool finite_gradient =
            std::isfinite(proposed.gradient_infinity) && std::isfinite(proposed.gradient_threshold);
        finite = finite_weights && finite_gradient;
        // CUDA also rejects nonfinite epoch differences. Preserve the old
        // disabled CPU path while enforcing the same guard arithmetic policy.
        if (backoff)
          finite = finite && std::isfinite(difference) && std::isfinite(proposed.intercept_change);
        proposed.reason = !finite_weights ? "nonfinite_weights" : "nonfinite_gradient";
        proposed.converged = finite && proposed.gradient_infinity <= proposed.gradient_threshold;
        if (!finite || proposed.converged)
          break;
      }
      double objective = guard ? guard->objective(state.weights) : nan;
      if (guard)
        ++result.probe_checks;
      if (guard && guard->rejects(objective, finite)) {
        result.rejected_epochs += done;
        // CPU updates mutate these buffers, unlike the CUDA fit operator.
        // Restore every state field before either retrying or returning.
        copy_state(state, *checkpoint);
        sample_rng = sampler_checkpoint;
        if (result.backoffs >= retry_limit) {
          terminal_numerical = true;
          break;
        }
        if (result.iterations >= epochs)
          break;
        ++result.backoffs;
        rate *= .5;
        continue;
      }
      result.accepted_epochs += done;
      result.accepted_updates += done * steps;
      result.accepted_rate = rate;
      result.maximum_coefficient = proposed.maximum_coefficient;
      result.maximum_change = proposed.maximum_change;
      result.intercept_change = proposed.intercept_change;
      result.relative_change = proposed.relative_change;
      result.gradient_infinity = proposed.gradient_infinity;
      result.gradient_threshold = proposed.gradient_threshold;
      result.reason = proposed.reason;
      result.converged = proposed.converged;
      terminal_numerical = !finite;
      if (guard) {
        guard->previous = objective;
        result.probe_objective = objective;
      }
      accepted = true;
      break;
    }
    if (result.converged || terminal_numerical || !accepted)
      break;
  }
  result.attempt_rate = rate;
  if (!result.accepted_updates)
    terminal_numerical = true;
  result.reason =
      result.converged
          ? "online_gradient"
          : (terminal_numerical ? (backoff ? "numerical_failure" : result.reason) : "max_epochs");
  result.coefficients = state.weights;
  result.gradient_squared =
      result.accepted_updates ? at::dot(state.gradient, state.gradient).item<double>() : nan;
  return result;
}
// Exposes the same production update to independent dense mathematical
// oracles, including nonzero cached state. This operator does not fit a
// preconditioner, allowing a prescribed factor and prescribed unique batches.
std::vector<Tensor> prescribed_updates(const Design& x, const Tensor& labels,
                                       const Tensor& schedule, const Tensor& weights,
                                       const Tensor& table, const Tensor& average,
                                       const Tensor& root, const Tensor& basis,
                                       const Tensor& ratios, double ridge, double rate) {
  valid(x);
  TORCH_CHECK(schedule.device().is_cpu() && schedule.scalar_type() == at::kLong &&
                  schedule.dim() == 2 && schedule.is_contiguous() && schedule.size(1) > 0,
              "schedule must be CPU int64[updates,batch]");
  for (const auto& item : std::vector<std::pair<Tensor, int64_t>>{
           {labels, x.n()}, {weights, x.d()}, {table, x.n()}, {average, x.d()}})
    TORCH_CHECK(item.first.device().is_cpu() &&
                    item.first.scalar_type() == x.numbers.scalar_type() && item.first.dim() == 1 &&
                    item.first.is_contiguous() && item.first.numel() == item.second,
                "invalid prescribed SAGA state");
  TORCH_CHECK(std::isfinite(ridge) && ridge >= 0 && std::isfinite(rate) && rate > 0,
              "invalid update parameters");
  std::unordered_set<int64_t> unique;
  for (int64_t i = 0; i < schedule.size(0); ++i) {
    unique.clear();
    for (int64_t j = 0; j < schedule.size(1); ++j) {
      auto r = schedule.const_data_ptr<int64_t>()[i * schedule.size(1) + j];
      TORCH_CHECK(r >= 0 && r < x.n() && unique.insert(r).second,
                  "minibatch rows must be unique and in bounds");
    }
  }
  State state{weights.clone(), table.clone(), average.clone(), at::zeros_like(weights),
              weights.clone()};
  std::optional<Inverse> p;
  if (root.numel()) {
    TORCH_CHECK(root.device().is_cpu() && root.scalar_type() == x.numbers.scalar_type() &&
                    root.dim() == 1 && root.numel() == x.d() && basis.device().is_cpu() &&
                    basis.scalar_type() == x.numbers.scalar_type() && basis.dim() == 2 &&
                    basis.size(0) == x.d() && ratios.device().is_cpu() &&
                    ratios.scalar_type() == x.numbers.scalar_type() && ratios.dim() == 1 &&
                    ratios.numel() == basis.size(1),
                "invalid prescribed inverse factors");
    p.emplace();
    p->root = root;
    p->basis = basis;
    p->ratios = ratios;
  }
  for (int64_t i = 0; i < schedule.size(0); ++i)
    update(x, labels, schedule.select(0, i), state, ridge, rate, p ? &*p : nullptr);
  auto [magnitude, difference] = change(state);
  auto diagnostic = at::tensor({magnitude, difference,
                                magnitude ? difference / magnitude : (difference ? INFINITY : 0),
                                at::dot(state.gradient, state.gradient).item<double>()},
                               weights.options());
  return {state.weights, state.table, state.average, state.gradient, diagnostic};
}

} // namespace ltm::saga::detail
