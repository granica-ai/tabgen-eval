#pragma once
#include <ATen/ATen.h>
#include <limits>

namespace ltm {
using at::Tensor;
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();

} // namespace ltm
