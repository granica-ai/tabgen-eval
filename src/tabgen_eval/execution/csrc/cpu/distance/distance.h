#pragma once
#include "cpu/runtime/sample.h"
#include <vector>

namespace ltm {
/** @brief Return four PRDC scores from full exact tiled pairwise work; complete
 * rows define compact geometry, self is rank zero and radius membership is
 * strict. */
template <class T>
std::vector<double> prdc(const Sample<T>& a, const Sample<T>& b, int64_t k, int threads,
                         int64_t components = 15);

} // namespace ltm
