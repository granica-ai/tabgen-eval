#pragma once
#include "cpu/runtime/types.h"

namespace ltm {
Tensor gram(const Tensor& a, const Tensor& b);
Tensor pair_counts(const Tensor& x);
Tensor keys(const Tensor& c, const Tensor& pairs, const Tensor& widths);
Tensor dense(const Tensor& c, const Tensor& pairs, const Tensor& widths, int64_t grid);
Tensor sorted_tv(const Tensor& a, const Tensor& b, const Tensor& na, const Tensor& nb,
                 const Tensor& proto);
bool rare(const Tensor& x, const Tensor& c, int64_t minimum, int64_t bins, double threshold);

} // namespace ltm
