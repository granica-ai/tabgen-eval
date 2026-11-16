#include "common/preprocessing/numeric.hpp"
#include "cpu/preprocessing/operators.h"
#include "cpu/runtime/tensor_validation.h"
#include <ATen/Parallel.h>

namespace ltm::preprocessing {
namespace {
void vector(const Tensor& t, at::ScalarType dtype) {
  TORCH_CHECK(t.device().is_cpu() && t.dim() == 1 && t.is_contiguous() && t.scalar_type() == dtype,
              "preprocessing state must be contiguous CPU vectors with matching dtype");
}
} // namespace
// Column reductions avoid allocating a full elementwise missingness matrix.
Tensor count_missing(const Tensor& values, int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  numbers(values);
  auto out = at::empty({values.size(1)}, values.options().dtype(at::kLong));
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "preprocessing_count_missing", [&] {
    auto* src = values.const_data_ptr<scalar_t>();
    auto* counts = out.data_ptr<int64_t>();
    at::parallel_for(0, values.size(1), 1, [&](int64_t begin, int64_t end) {
      for (int64_t c = begin; c < end; ++c) {
        int64_t missing = 0;
        for (int64_t r = 0; r < values.size(0); ++r)
          missing += !std::isfinite(src[r * values.size(1) + c]);
        counts[c] = missing;
      }
    });
  });
  return out;
}

// Dense consumers share one assembly boundary; implicit solvers need not call it.
Tensor assemble(const Tensor& numeric, const Tensor& codes, int64_t feature_count, bool linear,
                int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  numbers(numeric);
  matrix(codes, at::kLong);
  TORCH_CHECK(numeric.size(0) == codes.size(0) && feature_count >= 0,
              "numeric/category rows must match and feature count must be nonnegative");
  int64_t category_width = linear ? feature_count : codes.size(1);
  int64_t width = numeric.size(1) + category_width;
  auto out = at::empty({numeric.size(0), width}, numeric.options());
  AT_DISPATCH_FLOATING_TYPES(numeric.scalar_type(), "preprocessing_assemble", [&] {
    auto* dst = out.data_ptr<scalar_t>();
    auto* src = numeric.const_data_ptr<scalar_t>();
    auto* identities = codes.const_data_ptr<int64_t>();
    at::parallel_for(0, numeric.size(0), 1024, [&](int64_t begin, int64_t end) {
      for (int64_t r = begin; r < end; ++r) {
        auto* row = dst + r * width;
        std::copy_n(src + r * numeric.size(1), numeric.size(1), row);
        if (linear)
          std::fill_n(row + numeric.size(1), category_width, scalar_t(0));
        for (int64_t c = 0; c < codes.size(1); ++c) {
          int64_t id = identities[r * codes.size(1) + c];
          TORCH_CHECK(id >= -1 && (!linear || id < feature_count),
                      "category feature outside design");
          if (linear) {
            if (id >= 0)
              row[numeric.size(1) + id] = 1;
          } else
            row[numeric.size(1) + c] = scalar_t(id);
        }
      }
    });
  });
  return out;
}

// Fuse source-precision matching and feature-major/row-major layout conversion.
Tensor quantize(const Tensor& values, const Tensor& bits, at::ScalarType dtype, int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  TORCH_CHECK(values.device().is_cpu() && values.dim() == 2 &&
                  (values.scalar_type() == at::kFloat || values.scalar_type() == at::kDouble) &&
                  (dtype == at::kFloat || dtype == at::kDouble),
              "numeric quantization requires CPU float32/float64 matrices and output dtype");
  vector(bits, at::kLong);
  TORCH_CHECK(bits.numel() == values.size(1), "precision widths must match numeric columns");
  for (int64_t c = 0; c < bits.numel(); ++c) {
    int64_t b = bits.const_data_ptr<int64_t>()[c];
    TORCH_CHECK(b == 0 || b == 16 || b == 32 || b == 64,
                "numeric observation precision must be 0, 16, 32, or 64 bits");
  }
  auto out = at::empty(values.sizes(), values.options().dtype(dtype));
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "preprocessing_quantize_source", [&] {
    auto* src = values.const_data_ptr<scalar_t>();
    auto apply = [&](auto output) {
      using output_t = decltype(output);
      auto* dst = out.data_ptr<output_t>();
      at::parallel_for(0, values.size(0), 1024, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r)
          for (int64_t c = 0; c < values.size(1); ++c) {
            double v = src[r * values.stride(0) + c * values.stride(1)];
            auto width = bits.const_data_ptr<int64_t>()[c];
            if (width == 16)
              v = float(at::Half(float(v)));
            else if (width == 32)
              v = float(v);
            dst[r * values.size(1) + c] = output_t(v);
          }
      });
    };
    if (dtype == at::kDouble)
      apply(double{});
    else
      apply(float{});
  });
  return out;
}

std::vector<Tensor> fit_numeric(const Tensor& reference, const Tensor& candidate, bool linear,
                                int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  numbers(reference);
  numbers(candidate);
  TORCH_CHECK(reference.scalar_type() == candidate.scalar_type() &&
                  reference.size(1) == candidate.size(1) && reference.size(0) > 0 &&
                  candidate.size(0) > 0,
              "training numeric matrices must have matching columns/dtype and nonempty classes");
  int64_t width = reference.size(1);
  std::vector<NumericColumn> state(width);
  AT_DISPATCH_FLOATING_TYPES(reference.scalar_type(), "preprocessing_fit_numeric", [&] {
    auto* r = reference.const_data_ptr<scalar_t>();
    auto* s = candidate.const_data_ptr<scalar_t>();
    at::parallel_for(0, width, 1, [&](int64_t begin, int64_t end) {
      for (int64_t c = begin; c < end; ++c)
        state[c] = fit_numeric_column(
            reference.size(0), candidate.size(0),
            [&](int g, int64_t row) { return (g ? s : r)[row * width + c]; }, linear);
    });
  });
  std::vector<int64_t> kept, flags;
  for (int64_t c = 0; c < width; ++c)
    if (state[c].keep) {
      kept.push_back(c);
      if (state[c].indicator)
        flags.push_back(c);
    }
  auto ids = reference.options().dtype(at::kLong);
  auto keep = at::empty({int64_t(kept.size())}, ids);
  auto indicators = at::empty({int64_t(flags.size())}, ids);
  std::copy(kept.begin(), kept.end(), keep.data_ptr<int64_t>());
  std::copy(flags.begin(), flags.end(), indicators.data_ptr<int64_t>());
  auto impute = at::empty({int64_t(kept.size())}, reference.options());
  auto center = at::empty_like(impute), scale = at::empty_like(impute);
  AT_DISPATCH_FLOATING_TYPES(reference.scalar_type(), "preprocessing_numeric_state", [&] {
    for (size_t c = 0; c < kept.size(); ++c) {
      impute.data_ptr<scalar_t>()[c] = state[kept[c]].impute;
      center.data_ptr<scalar_t>()[c] = state[kept[c]].center;
      scale.data_ptr<scalar_t>()[c] = state[kept[c]].scale;
    }
  });
  return {keep, indicators, impute, center, scale};
}

Tensor apply_numeric(const Tensor& values, const Tensor& keep, const Tensor& indicators,
                     const Tensor& impute, const Tensor& center, const Tensor& scale, bool linear,
                     int64_t threads) {
  ThreadBudget budget(checked_threads(threads));
  numbers(values);
  vector(keep, at::kLong);
  vector(indicators, at::kLong);
  for (const auto& t : {impute, center, scale})
    vector(t, values.scalar_type());
  TORCH_CHECK(impute.numel() == keep.numel() && center.numel() == keep.numel() &&
                  scale.numel() == keep.numel() && (linear || indicators.numel() == 0),
              "inconsistent numeric preprocessing state");
  for (const auto& t : {keep, indicators})
    for (int64_t i = 0; i < t.numel(); ++i)
      TORCH_CHECK(t.const_data_ptr<int64_t>()[i] >= 0 &&
                      t.const_data_ptr<int64_t>()[i] < values.size(1),
                  "numeric preprocessing column outside input");
  int64_t width = keep.numel() + indicators.numel();
  auto out = at::empty({values.size(0), width}, values.options());
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "preprocessing_apply_numeric", [&] {
    auto* src = values.const_data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    auto* columns = keep.const_data_ptr<int64_t>();
    auto* flags = indicators.const_data_ptr<int64_t>();
    auto* fill = impute.const_data_ptr<scalar_t>();
    auto* means = center.const_data_ptr<scalar_t>();
    auto* scales = scale.const_data_ptr<scalar_t>();
    at::parallel_for(0, values.size(0), 1024, [&](int64_t begin, int64_t end) {
      for (int64_t r = begin; r < end; ++r) {
        for (int64_t c = 0; c < keep.numel(); ++c) {
          scalar_t v = src[r * values.size(1) + columns[c]];
          dst[r * width + c] =
              linear ? (std::isfinite(v) ? v - means[c] : fill[c] - means[c]) / scales[c]
                     : (std::isfinite(v) ? v : std::numeric_limits<scalar_t>::quiet_NaN());
        }
        for (int64_t c = 0; c < indicators.numel(); ++c)
          dst[r * width + keep.numel() + c] = !std::isfinite(src[r * values.size(1) + flags[c]]);
      }
    });
  });
  return out;
}
} // namespace ltm::preprocessing
