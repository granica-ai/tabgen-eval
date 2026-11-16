#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ltm::cuda::classifier::preparation {
std::vector<at::Tensor> prepare(const std::vector<at::Tensor>& inputs, const std::string& ptx,
                                double ridge, int64_t rank,
                                const std::optional<at::Generator>& generator, int64_t limit,
                                bool sampled, int64_t preparation_rows, int64_t batch);
std::vector<at::Tensor> factor(const at::Tensor& diagonal, const at::Tensor& omega,
                               const at::Tensor& images, double shift, int64_t limit);
int64_t preparation_policy(int64_t n, bool enabled, int64_t requested);
at::Tensor preparation_sample(int64_t n, int64_t count, int64_t seed);
} // namespace ltm::cuda::classifier::preparation
