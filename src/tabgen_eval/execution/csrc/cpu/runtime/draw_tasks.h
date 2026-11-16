#pragma once
#include "cpu/runtime/types.h"
#include <ATen/core/Dict.h>
#include <string>
#include <vector>

namespace ltm {
Tensor samples(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
               const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
               std::vector<int64_t> cards, std::string metric, int64_t k, int64_t minimum,
               int64_t bins, double rare, int64_t threads, int64_t components);
/** @brief Compute one point and collect its descriptive fidelity tables in the same pass. */
c10::Dict<std::string, Tensor>
fidelity_details(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc,
                 const Tensor& rm, const Tensor& sm, const Tensor& rd, const Tensor& sd,
                 std::vector<int64_t> cards, std::string metric, int64_t minimum, int64_t bins,
                 double rare, int64_t threads, int64_t components);

} // namespace ltm
