#pragma once
#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

// Shared CPU/CUDA policy. Only curvature preparation is sampled; solver
// observations, category transforms, ridge normalization and caches stay full.
namespace ltm_saga_preparation {
constexpr int64_t threshold = 500000;
constexpr uint64_t sampling_domain = 0xC0A7;
inline int64_t rows(int64_t n, bool enabled, int64_t requested) {
  return enabled && n >= threshold ? std::min(n, requested) : n;
}
inline std::vector<int64_t> sample(int64_t n, int64_t count, uint64_t seed) {
  std::mt19937_64 rng(seed ^ sampling_domain);
  std::unordered_set<int64_t> chosen;
  chosen.reserve(count * 2);
  std::vector<int64_t> result;
  result.reserve(count);
  // Floyd sampling consumes O(count) memory and never advances another RNG.
  for (int64_t j = n - count; j < n; ++j) {
    auto k = std::uniform_int_distribution<int64_t>(0, j)(rng);
    auto value = chosen.count(k) ? j : k;
    chosen.insert(value);
    result.push_back(value);
  }
  return result;
}
inline int64_t hash(const std::vector<int64_t>& rows) {
  uint64_t value = 1469598103934665603ULL;
  for (auto row : rows) {
    value ^= uint64_t(row);
    value *= 1099511628211ULL;
  }
  // Exactly representable in the existing floating diagnostic ABI.
  return value & ((1ULL << 53) - 1);
}
} // namespace ltm_saga_preparation
