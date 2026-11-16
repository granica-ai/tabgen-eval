#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ltm::cuda::classifier::saga {
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t,
                          int64_t, at::Tensor>;
bool setup(const std::string& path);
// tolerance bounds the online gradient infinity norm; initial_norm is diagnostic only.
Result fit(const std::vector<at::Tensor>& inputs, const std::string& ptx, double rate, double ridge,
           double tolerance, int64_t inverse_mode, bool stopping, bool averaged, int64_t limit,
           double initial_norm);
} // namespace ltm::cuda::classifier::saga
