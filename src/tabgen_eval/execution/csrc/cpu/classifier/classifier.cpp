#include "cpu/classifier/detail/newton.h"
#include "cpu/classifier/diagnostics.h"
#include "cpu/classifier/operators.h"
#include "cpu/runtime/tensor_validation.h"
#include "cpu/runtime/thread_budget.h"
#include <exception>
#include <new>

namespace ltm {
namespace {
using classifier_detail::Options;
void validate(const Options& o) {
  TORCH_CHECK(o.solver == "auto" || o.solver == "direct" || o.solver == "sketchysaga",
              "solver must be auto, direct, or sketchysaga");
  TORCH_CHECK(std::isfinite(o.ridge) && o.ridge >= 0 && std::isfinite(o.tolerance) &&
                  o.tolerance > 0 && o.max_iter > 0 && o.direct_columns > 0 &&
                  std::isfinite(o.direct_work) && o.direct_work > 0 && o.saga_epochs > 0 &&
                  o.saga_batch > 0 && o.saga_rank > 0 && o.saga_preparation_rows > 0,
              "invalid classifier options");
}
void validate_design(const Tensor& x, const Tensor& y) {
  numbers(x);
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 1 && y.device().is_cpu() && y.dim() == 1 &&
                  y.scalar_type() == x.scalar_type() && y.is_contiguous() && y.numel() == x.size(0),
              "invalid classifier design");
  TORCH_CHECK(at::isfinite(x).all().item<bool>() && at::isfinite(y).all().item<bool>() &&
                  ((y == 0) | (y == 1)).all().item<bool>() &&
                  (x.select(1, x.size(1) - 1) == 1).all().item<bool>(),
              "design must be finite with binary labels and a final intercept");
}

} // namespace
using classifier_detail::auc;
using classifier_detail::derivatives;
using classifier_detail::expand;
using classifier_detail::fit;
using classifier_detail::objective;
using classifier_detail::Options;
namespace fold_diagnostic = classifier_diagnostics::fold;
std::tuple<Tensor, Tensor, std::string, std::string>
classifier_fit(const Tensor& x, const Tensor& y, std::string solver, double ridge, double tolerance,
               int64_t max_iter, int64_t direct_columns, double direct_work, int64_t saga_epochs,
               int64_t saga_batch, int64_t saga_rank, int64_t seed, int64_t threads,
               bool saga_backoff, bool saga_sampled_preparation, int64_t saga_preparation_rows) {
  Options o{solver,
            ridge,
            tolerance,
            max_iter,
            direct_columns,
            direct_work,
            saga_epochs,
            saga_batch,
            saga_rank,
            saga_backoff,
            saga_sampled_preparation,
            saga_preparation_rows};
  validate(o);
  validate_design(x, y);
  TORCH_CHECK(threads > 0 && threads <= INT32_MAX, "invalid fit thread budget");
  ThreadBudget budget(threads);
  o.ridge = o.ridge ? o.ridge : 1. / x.size(0);
  bool direct = solver == "direct" || (solver == "auto" && x.size(1) - 1 <= direct_columns &&
                                       double(x.size(0)) * x.size(1) * x.size(1) <= direct_work);
  if (direct) {
    auto r = fit(x, y, o);
    auto diag = at::tensor(
        {double(r.converged), double(r.iterations), 0., r.gradient_squared, r.decrement_squared},
        x.options());
    return {r.coefficients, diag, r.solver, r.reason};
  }
  saga::Design design{x.narrow(1, 0, x.size(1) - 1).contiguous(),
                      at::empty({x.size(0), 0}, x.options().dtype(at::kLong)),
                      at::empty({0}, x.options()), at::empty({0}, x.options())};
  auto r =
      saga::fit(design, y, o.ridge, tolerance, saga_epochs, saga_batch, saga_rank, uint64_t(seed),
                INT64_MAX, saga_backoff, saga_sampled_preparation, saga_preparation_rows);
  std::vector<double> values = {double(r.converged), double(r.updates), double(r.iterations),
                                r.gradient_squared, r.relative_change};
  auto guard = saga::guard_diagnostics(r);
  values.insert(values.end(), guard.begin(), guard.end());
  auto preparation = saga::preparation_diagnostics(r);
  values.insert(values.end(), preparation.begin(), preparation.end());
  auto diag = at::tensor(values, x.options());
  return {r.coefficients, diag, "sketchysaga", r.reason};
}

std::tuple<Tensor, Tensor, Tensor> classifier_derivatives(const Tensor& x, const Tensor& y,
                                                          const Tensor& w, double ridge) {
  validate_design(x, y);
  TORCH_CHECK(w.device().is_cpu() && w.scalar_type() == x.scalar_type() && w.dim() == 1 &&
                  w.numel() == x.size(1) && w.is_contiguous() &&
                  at::isfinite(w).all().item<bool>() && std::isfinite(ridge) && ridge > 0,
              "invalid classifier coefficients/ridge");
  ThreadBudget budget(1);
  auto [gradient, curvature] = derivatives(x, y, w, ridge);
  return {at::scalar_tensor(objective(x, y, w, ridge), x.options()), gradient, curvature};
}

std::tuple<Tensor, Tensor, std::vector<std::string>, std::vector<std::string>,
           std::vector<std::vector<Tensor>>>
classifier(const Tensor& rn, const Tensor& rc, const Tensor& sn, const Tensor& sc, const Tensor& rm,
           const Tensor& sm, const Tensor& rd, const Tensor& sd, const Tensor& rf, const Tensor& sf,
           const Tensor& seeds, int64_t folds, std::string solver, double ridge, double tolerance,
           int64_t max_iter, int64_t direct_columns, double direct_work, int64_t saga_epochs,
           int64_t saga_batch, int64_t saga_rank, int64_t memory, int64_t workers,
           int64_t fit_threads, bool saga_backoff, bool saga_sampled_preparation,
           int64_t saga_preparation_rows) {
  Options o{solver,
            ridge,
            tolerance,
            max_iter,
            direct_columns,
            direct_work,
            saga_epochs,
            saga_batch,
            saga_rank,
            saga_backoff,
            saga_sampled_preparation,
            saga_preparation_rows};
  validate(o);
  validate_pool(rn, rc, rm);
  validate_pool(sn, sc, sm);
  TORCH_CHECK(rn.scalar_type() == sn.scalar_type(), "numeric pool dtypes must match");
  validate_draws(rd, rn.size(0));
  validate_draws(sd, sn.size(0));
  matrix(rf, at::kLong);
  matrix(sf, at::kLong);
  TORCH_CHECK(rd.size(0) == sd.size(0) && rf.sizes() == rd.sizes() && sf.sizes() == sd.sizes() &&
                  rn.size(1) == sn.size(1) && rc.size(1) == sc.size(1) && folds >= 2 &&
                  folds <= 16 && workers > 0 && workers <= INT32_MAX && fit_threads > 0 &&
                  fit_threads <= workers && memory > 0 && seeds.device().is_cpu() &&
                  seeds.scalar_type() == at::kLong && seeds.dim() == 1 && seeds.is_contiguous() &&
                  seeds.numel() == rd.size(0),
              "invalid classifier batches/options");
  for (auto labels : {rf, sf})
    for (int64_t i = 0; i < labels.numel(); ++i)
      TORCH_CHECK(labels.const_data_ptr<int64_t>()[i] >= 0 &&
                      labels.const_data_ptr<int64_t>()[i] < folds,
                  "fold label outside configured range");
  auto execute = [&](auto prototype) {
    using scalar_t = decltype(prototype);
    auto values = at::empty({rd.size(0)}, rn.options().dtype(at::kDouble));
    constexpr int64_t diagnostic_width = fold_diagnostic::width;
    auto diagnostics =
        at::full({rd.size(0), folds, diagnostic_width}, nan, rn.options().dtype(at::kDouble));
    diagnostics.narrow(2, 0, 6).zero_();
    std::vector<std::string> solvers(rd.size(0) * folds, "none"),
        reasons(rd.size(0) * folds, "preprocessing_or_memory_limit");
    std::vector<std::vector<Tensor>> preprocessing(rd.size(0) * folds);
    std::vector<std::exception_ptr> task_errors(rd.size(0));
    int64_t outer = std::max<int64_t>(1, std::min<int64_t>(rd.size(0), workers / fit_threads));
#pragma omp parallel for num_threads(outer) if (outer > 1)
    for (int64_t task = 0; task < rd.size(0); ++task) {
      try {
        ThreadBudget budget(fit_threads);
        long double owned_draw_bytes =
            (rd.size(1) + sd.size(1)) * (sizeof(scalar_t) * rn.size(1) + 8.L * rc.size(1) + 1);
        if (owned_draw_bytes > memory) {
          values.data_ptr<double>()[task] = nan;
          continue;
        }
        auto a = gather<scalar_t>(rn, rc, rm, rd.const_data_ptr<int64_t>() + task * rd.size(1),
                                  rd.size(1));
        auto b = gather<scalar_t>(sn, sc, sm, sd.const_data_ptr<int64_t>() + task * sd.size(1),
                                  sd.size(1));
        double total = 0;
        long double metadata_bytes = 0;
        for (int64_t fold = 0; fold < folds; ++fold) {
          int64_t index = task * folds + fold;
          auto* diag = diagnostics.data_ptr<double>() + index * diagnostic_width;
          diag[fold_diagnostic::training_rows] = 0;
          for (int64_t i = 0; i < a.n; ++i)
            diag[fold_diagnostic::training_rows] +=
                rf.const_data_ptr<int64_t>()[task * rf.size(1) + i] != fold;
          for (int64_t i = 0; i < b.n; ++i)
            diag[fold_diagnostic::training_rows] +=
                sf.const_data_ptr<int64_t>()[task * sf.size(1) + i] != fold;
          diag[fold_diagnostic::validation_rows] = a.n + b.n - diag[fold_diagnostic::training_rows];
          diag[fold_diagnostic::encoded_columns] = diag[fold_diagnostic::converged] =
              diag[fold_diagnostic::iterations] = diag[fold_diagnostic::epochs_completed] = 0;
          if (owned_draw_bytes + metadata_bytes >= memory) {
            total = nan;
            continue;
          }
          auto data = saga::prepare(a, b, rf.const_data_ptr<int64_t>() + task * rf.size(1),
                                    sf.const_data_ptr<int64_t>() + task * sf.size(1), fold,
                                    memory - int64_t(owned_draw_bytes + metadata_bytes));
          if (!data) {
            total = nan;
            continue;
          }
          preprocessing[index] = data->preprocessing;
          for (const auto& tensor : data->preprocessing)
            metadata_bytes += tensor.numel() * tensor.element_size();
          diag[fold_diagnostic::training_rows] = data->train.n();
          diag[fold_diagnostic::validation_rows] = data->test.n();
          diag[fold_diagnostic::encoded_columns] = data->train.d() - 1;
          diag[fold_diagnostic::converged] = 0;
          if (data->train.d() == 1) {
            total = nan;
            continue;
          }
          Options local = o;
          local.ridge = local.ridge ? local.ridge : 1. / data->train.n();
          diag[fold_diagnostic::effective_ridge] = local.ridge;
          int64_t d = data->train.d(), nt = data->train.n(), nv = data->test.n();
          bool preferred =
              o.solver == "direct" || (o.solver == "auto" && d - 1 <= o.direct_columns &&
                                       double(nt) * d * d <= o.direct_work);
          long double compact =
              sizeof(scalar_t) * ((nt + nv) * (data->train.pn() + 1) + 2.L * data->train.k()) +
              8.L * (nt + nv) * data->train.pc();
          long double dense = owned_draw_bytes + metadata_bytes + compact +
                              sizeof(scalar_t) * (2.L * (nt + nv) * d + 5.L * d * d);
          bool direct = preferred && dense <= memory;
          if (o.solver == "direct" && !direct) {
            total = nan;
            solvers[index] = "direct";
            reasons[index] = "direct_memory_limit";
            continue;
          }
          uint64_t seed = uint64_t(seeds.const_data_ptr<int64_t>()[task]) +
                          uint64_t(fold) * 0x9e3779b97f4a7c15ULL;
          Tensor coefficients;
          if (direct) {
            auto x = expand(data->train);
            auto result = fit(x, data->labels, local);
            coefficients = result.coefficients;
            diag[fold_diagnostic::converged] = result.converged;
            diag[fold_diagnostic::iterations] = result.iterations;
            diag[fold_diagnostic::epochs_completed] = 0;
            diag[fold_diagnostic::gradient_squared] = result.gradient_squared;
            diag[fold_diagnostic::decrement_squared] = result.decrement_squared;
            solvers[index] = "direct";
            reasons[index] = result.reason;
          } else {
            // Reserve scoring/draw storage that survives training before passing
            // the remaining workspace to the compact fit allocation guard.
            long double retained = sizeof(scalar_t) * nv * (data->test.pn() + 1) +
                                   8.L * nv * data->test.pc() + owned_draw_bytes + metadata_bytes;
            long double required =
                saga::workspace_bytes(data->train, o.saga_batch, o.saga_rank, o.saga_backoff,
                                      o.saga_sampled_preparation, o.saga_preparation_rows);
            if (required + retained > memory) {
              total = nan;
              reasons[index] = "saga_memory_limit";
              continue;
            }
            solvers[index] = "sketchysaga";
            saga::Fit result;
            try {
              result =
                  saga::fit(data->train, data->labels, local.ridge, o.tolerance, o.saga_epochs,
                            o.saga_batch, o.saga_rank, seed, memory - int64_t(retained),
                            o.saga_backoff, o.saga_sampled_preparation, o.saga_preparation_rows);
            } catch (const c10::OutOfMemoryError&) {
              reasons[index] = "memory_limit";
              total = nan;
              continue;
            } catch (const std::bad_alloc&) {
              reasons[index] = "memory_limit";
              total = nan;
              continue;
            } catch (const c10::LinAlgError&) {
              reasons[index] = "numerical_failure";
              total = nan;
              continue;
            } catch (const saga::NumericalFailure&) {
              reasons[index] = "numerical_failure";
              total = nan;
              continue;
            }
            coefficients = result.coefficients;
            diag[fold_diagnostic::converged] = result.converged;
            diag[fold_diagnostic::iterations] = result.updates;
            diag[fold_diagnostic::epochs_completed] = result.iterations;
            diag[fold_diagnostic::gradient_squared] = result.gradient_squared;
            diag[fold_diagnostic::relative_coefficient_change] = result.relative_change;
            diag[fold_diagnostic::preconditioner_rank] = result.rank;
            diag[fold_diagnostic::learning_rate] = result.rate;
            diag[fold_diagnostic::maximum_feature_coefficient] = result.maximum_coefficient;
            diag[fold_diagnostic::maximum_feature_change] = result.maximum_change;
            diag[fold_diagnostic::intercept_change] = result.intercept_change;
            diag[fold_diagnostic::stable_inverse] = result.stable_inverse;
            diag[fold_diagnostic::estimated_eigenvalue] = result.eigenvalue;
            diag[fold_diagnostic::online_gradient_infinity] = result.gradient_infinity;
            diag[fold_diagnostic::initial_online_gradient_infinity] =
                result.initial_gradient_infinity;
            diag[fold_diagnostic::gradient_stopping_threshold] = result.gradient_threshold;
            auto guard = saga::guard_diagnostics(result);
            std::copy(guard.begin(), guard.end(), diag + fold_diagnostic::accepted_epochs);
            auto preparation = saga::preparation_diagnostics(result);
            std::copy(preparation.begin(), preparation.end(),
                      diag + fold_diagnostic::preparation_training_rows);
            solvers[index] = "sketchysaga";
            reasons[index] = result.reason;
          }
          // Budget-limited SAGA fits retain their finite AUC; unusable terminal
          // failures do not define a trained model. Direct Newton preserves its
          // established contract of scoring retained finite partial iterates.
          bool scorable =
              direct || reasons[index] == "online_gradient" || reasons[index] == "max_epochs";
          if (!scorable || !at::isfinite(coefficients).all().item<bool>()) {
            total = nan;
            continue;
          }
          double value = auc(saga::mv(data->test, coefficients), data->scoring);
          total += std::isfinite(value) ? std::max(.5, value) : nan;
        }
        values.data_ptr<double>()[task] = total / folds;
      } catch (...) {
        // Preserve programming, shape and library errors rather than relabelling
        // them optimizer failures. Exceptions cannot cross an OpenMP region.
        task_errors[task] = std::current_exception();
        values.data_ptr<double>()[task] = nan;
      }
    }
    for (const auto& error : task_errors)
      if (error)
        std::rethrow_exception(error);
    return std::make_tuple(values, diagnostics, solvers, reasons, preprocessing);
  };
  return rn.scalar_type() == at::kDouble ? execute(double{}) : execute(float{});
}

} // namespace ltm
