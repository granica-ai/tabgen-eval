#pragma once
#include "cpu/runtime/types.h"
#include <string>
#include <tuple>
#include <vector>

namespace ltm {
/** @brief Fit a finite CPU float32/float64 logistic design with an unpenalized intercept.
 * x is contiguous [N,D] with a final column of ones; y is contiguous binary [N]
 * in the same dtype. Inputs remain read-only and thread settings are restored.
 * @return Owned coefficients [D], solver-specific diagnostics, solver name and reason.
 * The first five diagnostic slots are convergence, iterations/updates, epochs,
 * squared gradient, and Newton decrement/relative coefficient change. SAGA appends
 * the guard and preparation records declared in diagnostics.h.
 * @note ridge=0 selects 1/N. Invalid layouts/options raise; finite nonconverged
 * fits retain their coefficients and termination diagnostics.
 * tolerance bounds the full-gradient L2 norm for Newton and online-gradient
 * infinity norm for SAGA.
 */
std::tuple<Tensor, Tensor, std::string, std::string>
classifier_fit(const Tensor& x, const Tensor& y, std::string solver, double ridge, double tolerance,
               int64_t max_iter, int64_t direct_columns, double direct_work, int64_t saga_epochs,
               int64_t saga_batch, int64_t saga_rank, int64_t seed, int64_t threads,
               bool saga_backoff, bool saga_sampled_preparation, int64_t saga_preparation_rows);

/** @brief Evaluate average logistic loss, gradient and per-row curvature.
 * x/y follow classifier_fit's CPU layout; w is contiguous [D] in the same dtype.
 * ridge must be positive and excludes the final intercept coefficient.
 * @return Owned scalar objective, [D] gradient and [N] curvature; inputs are read-only.
 * Invalid layouts, nonfinite coefficients or invalid ridge raise.
 */
std::tuple<Tensor, Tensor, Tensor> classifier_derivatives(const Tensor& x, const Tensor& y,
                                                          const Tensor& w, double ridge);

/** @brief Score source-stable classifier draws with fold-local fitted transforms.
 * rn/sn are contiguous CPU float32/float64 [source_rows,numeric_columns]; rc/sc
 * are aligned int64 [source_rows,categorical_columns], rm/sm are bool row masks.
 * rd/sd are int64 [B,draw_rows]; rf/sf have the same shapes and source-stable fold
 * labels. seeds is contiguous int64 [B]. Numeric sources share one dtype.
 * @return Float64 scores [B], diagnostics [B,folds,46] (diagnostics.h), and solver
 * names/reasons and six-tensor preprocessing metadata flattened draw-major, fold-major.
 * @note Inputs are read-only; memory budgets draw/preprocessing/fit storage.
 * Worker and fit-thread budgets are bounded and calling-thread settings restored.
 * Invalid layouts/options raise; unavailable fits retain diagnostic reasons.
 */
std::tuple<Tensor, Tensor, std::vector<std::string>, std::vector<std::string>,
           std::vector<std::vector<Tensor>>>
classifier(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc, const Tensor& rm,
           const Tensor& sm, const Tensor& rd, const Tensor& sd, const Tensor& rf, const Tensor& sf,
           const Tensor& seeds, int64_t folds, std::string solver, double ridge, double tolerance,
           int64_t max_iter, int64_t direct_columns, double direct_work, int64_t saga_epochs,
           int64_t saga_batch, int64_t saga_rank, int64_t memory, int64_t workers,
           int64_t fit_threads, bool saga_backoff, bool saga_sampled_preparation,
           int64_t saga_preparation_rows);

} // namespace ltm
