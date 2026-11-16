#pragma once
#include "cpu/classifier/saga.h"
#include <random>

namespace ltm::saga::detail {
void valid(const Design& x);
Tensor adjoint(const Design& x, const Tensor& v);
Tensor hessian(const Design& x, const Tensor& v, double ridge);
Tensor diagonal(const Design& x, double ridge);
struct Inverse {
  Tensor root, basis, ratios, reflectors, tau, rotations;
  bool stable = false;
  Tensor apply(const Tensor& rhs) const {
    auto v = root * rhs;
    if (!stable)
      return root * (v + at::mv(basis, (ratios - 1) * at::mv(basis.t(), v)));
    auto coordinates = at::ormqr(reflectors, tau, v.unsqueeze(1), true, true);
    auto head = coordinates.narrow(0, 0, ratios.numel()).squeeze(1);
    head.copy_(at::mv(rotations, ratios * at::mv(rotations.t(), head)));
    return root * at::ormqr(reflectors, tau, coordinates, true, false).squeeze(1);
  }
};
struct State {
  Tensor weights, table, average, gradient, previous;
};
Tensor gaussian(int64_t n, int64_t k, const at::TensorOptions& opts, std::mt19937_64& rng);
Inverse preconditioner(const Design& x, double ridge, int64_t rank, std::mt19937_64& rng);
Fit run(const Design& x, const Tensor& labels, double ridge, double tolerance, int64_t epochs,
        int64_t batch, uint64_t seed, double rate, const Inverse* inverse, bool backoff,
        State& state, const Tensor& schedule = {});
std::vector<Tensor> prescribed_updates(const Design& x, const Tensor& labels,
                                       const Tensor& schedule, const Tensor& weights,
                                       const Tensor& table, const Tensor& average,
                                       const Tensor& root, const Tensor& basis,
                                       const Tensor& ratios, double ridge, double rate);
inline constexpr int64_t probe_limit = 4096, retry_limit = 4, chunk_epochs = 16;

} // namespace ltm::saga::detail
