#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <tuple>

namespace ltm::cuda::linalg {
at::Tensor mv(const at::Tensor& x, const at::Tensor& v, bool transpose);
at::Tensor gemm(const at::Tensor& a, const at::Tensor& b);
at::Tensor lower_gram(const at::Tensor& x);
int64_t workspace(const at::Tensor& x);
std::tuple<at::Tensor, at::Tensor> cholesky(const at::Tensor& x);
std::tuple<at::Tensor, at::Tensor> solve(const at::Tensor& factor, const at::Tensor& rhs);
} // namespace ltm::cuda::linalg
