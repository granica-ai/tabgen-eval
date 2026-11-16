#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <string>
#include <tuple>

namespace ltm::cuda::classifier::newton {
// The host diagnostics are [full gradient squared, decrement squared, maximum
// direction damping]. Status describes the terminal fit, not Cholesky info.
enum class Status : int64_t {
  max_iter = 0,
  gradient_squared = 1,
  nonfinite_gradient = 2,
  factorization_failed = 3,
  invalid_direction = 4,
  line_search_failed = 5,
};
using Result = std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>;
/** @brief Retain a compiled Newton PTX module on the current CUDA device.
 * Throws on missing/invalid PTX or CUDA setup failure; retains no input tensors. */
bool setup(const std::string& path);
/** @brief Fit a read-only column-major CUDA FP32/FP64 design[N,D], D>1,
 * with matching contiguous binary labels[N] and unpenalized final intercept.
 * Uses the caller stream, restores device/BLAS modes and owns all output/scratch.
 * Returns coefficients[D], CPU double diagnostics[3], terminal Status and accepted
 * step count. Gradient tolerance bounds the full-gradient L2 norm; ridge is mean-loss
 * feature penalty. Throws on invalid packets, CUDA/library errors or a byte limit
 * smaller than the queried workspace and all allocated CUDA buffers. */
Result fit(const at::Tensor& design, const at::Tensor& labels, const std::string& ptx, double ridge,
           double tolerance, int64_t max_iter, int64_t limit);
} // namespace ltm::cuda::classifier::newton
