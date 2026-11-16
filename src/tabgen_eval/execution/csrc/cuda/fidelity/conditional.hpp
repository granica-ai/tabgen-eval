#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <cstdint>
#include <vector>

namespace ltm::cuda::fidelity {
std::vector<at::Tensor> dense_tv_batches(const at::Tensor& a, const at::Tensor& b,
                                         const at::Tensor& pairs, const at::Tensor& levels,
                                         const at::Tensor& prototype, int64_t grid, int64_t batch,
                                         int64_t cond, at::IntArrayRef histogram,
                                         at::IntArrayRef reduction);
} // namespace ltm::cuda::fidelity
