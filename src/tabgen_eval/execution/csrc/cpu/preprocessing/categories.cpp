#include "common/preprocessing/categories.hpp"
#include "cpu/preprocessing/operators.h"
#include "cpu/runtime/tensor_validation.h"
#include <ATen/Parallel.h>
#include <limits>

namespace ltm::preprocessing {
namespace {
void ids(const Tensor& t) {
  TORCH_CHECK(t.device().is_cpu() && t.dim() == 1 && t.scalar_type() == at::kLong &&
                  t.is_contiguous(),
              "categorical preprocessing state must be contiguous CPU int64 vectors");
}
} // namespace
std::vector<Tensor> fit_categories(const Tensor& reference, const Tensor& candidate,
                                   const Tensor& cardinalities, int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  matrix(reference, at::kLong);
  matrix(candidate, at::kLong);
  ids(cardinalities);
  TORCH_CHECK(reference.size(1) == candidate.size(1) && cardinalities.numel() == reference.size(1),
              "categorical training columns and cardinalities must match");
  int64_t width = reference.size(1);
  auto domains = at::empty({width + 1}, cardinalities.options());
  auto* offsets = domains.data_ptr<int64_t>();
  offsets[0] = 0;
  for (int64_t c = 0; c < width; ++c) {
    int64_t count = cardinalities.const_data_ptr<int64_t>()[c];
    TORCH_CHECK(count >= 0 && count < std::numeric_limits<int64_t>::max() - offsets[c],
                "invalid categorical cardinality");
    offsets[c + 1] = offsets[c] + count + 1;
  }
  auto mapping = at::full({offsets[width]}, -1, cardinalities.options());
  std::vector<std::vector<int64_t>> levels(width);
  at::parallel_for(0, width, 1, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; ++c) {
      auto* map = mapping.data_ptr<int64_t>() + offsets[c];
      int64_t cardinality = cardinalities.const_data_ptr<int64_t>()[c];
      for (const auto& source : {reference, candidate})
        for (int64_t r = 0; r < source.size(0); ++r) {
          int64_t code = source.const_data_ptr<int64_t>()[r * width + c];
          TORCH_CHECK(code >= -1 && code < cardinality,
                      "category code outside declared cardinality");
          map[code + 1] = 0;
        }
      for (int64_t code = -1; code < cardinality; ++code)
        if (map[code + 1] == 0) {
          map[code + 1] = levels[c].size();
          levels[c].push_back(code);
        }
    }
  });
  auto features = at::empty({width + 1}, cardinalities.options());
  auto* fp = features.data_ptr<int64_t>();
  fp[0] = 0;
  for (int64_t c = 0; c < width; ++c)
    fp[c + 1] = fp[c] + levels[c].size();
  auto active = at::empty({fp[width]}, cardinalities.options());
  for (int64_t c = 0; c < width; ++c)
    for (size_t j = 0; j < levels[c].size(); ++j)
      active.data_ptr<int64_t>()[fp[c] + j] = offsets[c] + levels[c][j] + 1;
  return {mapping, domains, features, active};
}

Tensor map_categories(const Tensor& codes, const Tensor& mapping, const Tensor& domains,
                      const Tensor& features, bool linear, int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  matrix(codes, at::kLong);
  for (const auto& t : {mapping, domains, features})
    ids(t);
  TORCH_CHECK(domains.numel() == codes.size(1) + 1 && features.numel() == domains.numel() &&
                  domains[0].item<int64_t>() == 0 && features[0].item<int64_t>() == 0,
              "categorical preprocessing offsets must match source columns");
  auto* dp = domains.const_data_ptr<int64_t>();
  auto* fp = features.const_data_ptr<int64_t>();
  TORCH_CHECK(dp[codes.size(1)] == mapping.numel(), "categorical mapping domain mismatch");
  for (int64_t c = 0; c < codes.size(1); ++c)
    TORCH_CHECK(dp[c + 1] > dp[c] && fp[c + 1] >= fp[c] && fp[c + 1] - fp[c] <= dp[c + 1] - dp[c],
                "invalid categorical preprocessing offsets");
  auto out = at::empty_like(codes);
  at::parallel_for(0, codes.size(0), 1024, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r)
      for (int64_t c = 0; c < codes.size(1); ++c) {
        int64_t index = r * codes.size(1) + c;
        int64_t code = codes.const_data_ptr<int64_t>()[index];
        TORCH_CHECK(code >= -1 && code < dp[c + 1] - dp[c] - 1,
                    "category code outside declared cardinality");
        int64_t mapped = mapping.const_data_ptr<int64_t>()[dp[c] + code + 1];
        TORCH_CHECK(mapped >= -1 && mapped < fp[c + 1] - fp[c], "invalid category mapping");
        out.data_ptr<int64_t>()[index] =
            mapped < 0 ? (linear ? -1 : fp[c + 1] - fp[c]) : mapped + (linear ? fp[c] : 0);
      }
  });
  return out;
}
} // namespace ltm::preprocessing
