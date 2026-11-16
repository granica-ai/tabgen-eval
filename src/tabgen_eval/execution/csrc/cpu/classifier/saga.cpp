#include "cpu/classifier/detail/saga.h"
#include "common/classifier/saga_preparation.hpp"
#include "common/classifier/saga_rate.hpp"
#include "common/classifier/saga_row_curvature.hpp"

namespace ltm::saga {
using detail::diagonal;
using detail::gaussian;
using detail::hessian;
using detail::Inverse;
using detail::preconditioner;
using detail::run;
using detail::State;
using detail::valid;
Fit fit(const Design& x, const Tensor& labels, double ridge, double tolerance, int64_t epochs,
        int64_t batch_size, int64_t rank, uint64_t seed, int64_t memory, bool backoff, bool sampled,
        int64_t preparation_rows) {
  valid(x);
  TORCH_CHECK(labels.device().is_cpu() && labels.scalar_type() == x.numbers.scalar_type() &&
                  labels.is_contiguous() && labels.dim() == 1 && labels.numel() == x.n() &&
                  ((labels == 0) | (labels == 1)).all().item<bool>(),
              "invalid SAGA labels");
  TORCH_CHECK(std::isfinite(ridge) && ridge > 0 && std::isfinite(tolerance) && tolerance > 0 &&
                  epochs > 0 && batch_size > 0 && rank > 0 && memory > 0 && preparation_rows > 0,
              "invalid SAGA options");
  rank = std::min(rank, x.d());
  int64_t b = std::min(batch_size, x.n());
  long double bytes = workspace_bytes(x, b, rank, backoff, sampled, preparation_rows);
  TORCH_CHECK(bytes <= memory, "compact SAGA workspace exceeds model memory limit");
  std::mt19937_64 sketch_rng(seed ^ 0x5A6A);
  Design curvature = x;
  int64_t curvature_n = ltm_saga_preparation::rows(x.n(), sampled, preparation_rows);
  int64_t sample_hash = 0;
  if (curvature_n < x.n()) {
    auto selected = ltm_saga_preparation::sample(x.n(), curvature_n, seed ^ 0x5A6A);
    sample_hash = ltm_saga_preparation::hash(selected);
    auto rows = at::from_blob(selected.data(), {curvature_n}, x.identities.options()).clone();
    curvature.numbers = x.numbers.index_select(0, rows);
    curvature.identities = x.identities.index_select(0, rows);
  }
  Inverse inverse;
  try {
    inverse = preconditioner(curvature, ridge, rank, sketch_rng);
  } catch (const c10::LinAlgError&) {
    auto diag = diagonal(curvature, ridge);
    if (!at::isfinite(diag).all().item<bool>() || !(diag > 0).all().item<bool>())
      throw NumericalFailure("invalid initial Hessian diagonal");
    inverse.root = diag.rsqrt();
    inverse.basis = at::empty({x.d(), 0}, x.numbers.options());
    inverse.ratios = at::empty({0}, x.numbers.options());
    rank = 0;
  }
  auto v = gaussian(x.d(), 1, x.numbers.options(), sketch_rng).select(1, 0);
  v /= at::norm(v);
  double lambda = 0;
  for (int i = 0; i < 30; ++i) {
    auto h = hessian(curvature, v, 0);
    auto next = inverse.apply(h);
    double norm = at::norm(next).item<double>();
    if (!std::isfinite(norm) || norm <= 0)
      throw NumericalFailure("invalid preconditioned curvature");
    lambda = at::dot(h, next).item<double>() / at::dot(v, h).item<double>();
    v = next / norm;
  }
  if (!std::isfinite(lambda) || lambda <= 0)
    throw NumericalFailure("invalid SAGA curvature estimate");
  Design calibration_design = curvature;
  int64_t calibration_n = std::min<int64_t>(4096, curvature_n);
  if (calibration_n < curvature_n) {
    auto selected =
        ltm_saga_preparation::sample(curvature_n, calibration_n, (seed ^ 0x5A6A) ^ 0xCA11);
    auto rows = at::from_blob(selected.data(), {calibration_n}, x.identities.options()).clone();
    calibration_design.numbers = curvature.numbers.index_select(0, rows);
    calibration_design.identities = curvature.identities.index_select(0, rows);
  }
  auto row_curvatures = calibration::row_curvature(
      calibration_design.numbers, calibration_design.identities, calibration_design.centers,
      calibration_design.scales, inverse.root, inverse.basis, inverse.ratios);
  auto calibrated =
      calibration::rate(x.n(), b, calibration_n, lambda, row_curvatures.max().item<double>(),
                        calibration::ridge_bound(inverse.root, ridge));
  double rate = calibrated.rate;
  State state{at::zeros({x.d()}, x.numbers.options()), at::zeros({x.n()}, x.numbers.options()),
              at::zeros({x.d()}, x.numbers.options()), at::zeros({x.d()}, x.numbers.options()),
              at::zeros({x.d()}, x.numbers.options())};
  auto result = run(x, labels, ridge, tolerance, epochs, b, seed, rate, &inverse, backoff, state);
  result.rank = rank;
  result.eigenvalue = calibrated.data_used;
  result.stable_inverse = inverse.stable;
  result.preparation_training_rows = x.n();
  result.preparation_rows = curvature_n;
  result.sampled_preparation = curvature_n < x.n();
  result.preparation_sample_hash = sample_hash;
  result.rate_calibration = {
      calibrated.data_raw,     calibrated.data_used,          calibrated.maximum_raw,
      calibrated.maximum_used, calibrated.ridge_upper,        calibrated.practical,
      double(calibrated.rows), double(calibrated.population), double(calibrated.batch)};
  return result;
}

} // namespace ltm::saga
