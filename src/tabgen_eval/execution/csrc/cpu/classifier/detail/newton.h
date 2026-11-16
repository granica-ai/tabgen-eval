#pragma once
#include "cpu/classifier/saga.h"
#include <utility>

namespace ltm::classifier_detail {
struct Options {
  std::string solver;
  double ridge, tolerance;
  int64_t max_iter, direct_columns;
  double direct_work;
  int64_t saga_epochs, saga_batch, saga_rank;
  bool saga_backoff, saga_sampled_preparation;
  int64_t saga_preparation_rows;
};
struct Fit {
  Tensor coefficients;
  std::string solver, reason = "max_iter";
  bool converged = false;
  int64_t iterations = 0;
  double gradient_squared = nan, decrement_squared = nan;
};

double objective(const Tensor& x, const Tensor& y, const Tensor& w, double ridge);
std::pair<Tensor, Tensor> derivatives(const Tensor& x, const Tensor& y, const Tensor& w,
                                      double ridge);
Fit fit(const Tensor& x, const Tensor& y, const Options& o);
Tensor expand(const saga::Design& x);
double auc(const Tensor& scores, const Tensor& labels);

} // namespace ltm::classifier_detail
