#pragma once
#include "cpu/runtime/types.h"

namespace ltm {
Tensor distances(const Tensor& a, const Tensor& b, const Tensor& ca, const Tensor& cb);

} // namespace ltm
