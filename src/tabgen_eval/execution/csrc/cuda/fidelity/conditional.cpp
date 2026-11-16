// Experimental native orchestration of existing Triton histogram/TV kernels.
// Kernel modules are loaded and retained by native_tv.py; there are no Python
// callbacks in this loop. All allocation and launches use the caller's stream.
#include "cuda/fidelity/conditional.hpp"
#include "cuda/runtime/borrowed_kernel.hpp"
#include <ATen/ATen.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <cuda.h>
#include <utility>
#include <vector>

namespace ltm::cuda::fidelity {
std::vector<at::Tensor> dense_tv_batches(const at::Tensor& a, const at::Tensor& b,
                                         const at::Tensor& pairs, const at::Tensor& levels,
                                         const at::Tensor& prototype, int64_t grid, int64_t batch,
                                         int64_t cond, at::IntArrayRef histogram,
                                         at::IntArrayRef reduction) {
  for (const auto& value : {a, b, pairs, levels, prototype})
    TORCH_CHECK(value.is_cuda() && value.device() == a.device() && value.is_contiguous(),
                "Conditional native TV requires contiguous same-device CUDA tensors");
  for (const auto& value : {a, b, pairs, levels}) {
    TORCH_CHECK(value.scalar_type() == at::kLong,
                "Conditional native TV requires int64 codes/pairs/levels");
    // The warmup kernels assume the alignment of draw-owned allocations.
    // Reject offset views rather than silently using an incompatible binary.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(value.const_data_ptr()) % 16 == 0,
                "Conditional native TV requires 16-byte aligned inputs");
  }
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) && a.size(1) == b.size(1) &&
                  pairs.dim() == 2 && pairs.size(1) == 2 && levels.dim() == 2 &&
                  levels.size(0) == a.size(0) && levels.size(1) == a.size(1),
              "invalid Conditional native TV dimensions");
  TORCH_CHECK(prototype.scalar_type() == at::kFloat || prototype.scalar_type() == at::kDouble,
              "Conditional native TV requires FP32 or FP64 output");
  TORCH_CHECK(grid > 0 && grid <= 8192 && !(grid & (grid - 1)) && batch > 0 && batch <= 1024 &&
                  a.size(0) <= 65535 && a.size(1) <= INT_MAX && a.size(2) <= INT_MAX &&
                  b.size(2) <= INT_MAX && cond >= -1 && cond < a.size(1),
              "Conditional native TV exceeds supported launch bounds");
  c10::cuda::CUDAGuard guard(a.device());
  if (!a.size(0) || !pairs.size(0))
    return {at::empty({a.size(0), pairs.size(0)}, prototype.options())};
  runtime::BorrowedKernel hist(histogram), tv(reduction);
  auto stream =
      reinterpret_cast<CUstream>(c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
  int p = a.size(1), conditioning = cond;
  CUdeviceptr widths = reinterpret_cast<CUdeviceptr>(levels.const_data_ptr());
  // Triton appends global/profile scratch pointers even when their sizes are
  // zero. The Python bridge rejects kernels that actually require scratch.
  CUdeviceptr scratch = 0;
  std::vector<at::Tensor> outputs;
  for (int64_t start = 0; start < pairs.size(0); start += batch) {
    int q = std::min(batch, pairs.size(0) - start);
    auto ar = at::zeros({a.size(0), q, grid}, a.options());
    auto bs = at::zeros({a.size(0), q, grid}, b.options());
    auto out = at::empty({a.size(0), q}, prototype.options());
    CUdeviceptr selected =
        reinterpret_cast<CUdeviceptr>(pairs.const_data_ptr<int64_t>() + 2 * start);
    CUdeviceptr real = reinterpret_cast<CUdeviceptr>(ar.data_ptr());
    CUdeviceptr generated = reinterpret_cast<CUdeviceptr>(bs.data_ptr());
    CUdeviceptr scores = reinterpret_cast<CUdeviceptr>(out.data_ptr());
    for (const auto& item : {std::make_pair(a, real), std::make_pair(b, generated)}) {
      CUdeviceptr codes = reinterpret_cast<CUdeviceptr>(item.first.const_data_ptr());
      CUdeviceptr counts = item.second;
      int n = item.first.size(2);
      void* args[] = {&codes, &selected, &widths,       &counts,  &n,
                      &p,     &q,        &conditioning, &scratch, &scratch};
      int64_t row_blocks = (int64_t(n) + 1023) / 1024;
      hist.run(row_blocks * (grid > 4096 ? 2 : 1), q, a.size(0), stream, args);
    }
    void* args[] = {&real, &generated, &selected,     &widths,  &scores,
                    &q,    &p,         &conditioning, &scratch, &scratch};
    tv.run(q, a.size(0), 1, stream, args);
    outputs.push_back(out);
  }
  // The caller concatenates all grid buckets once, matching the Python path's
  // allocation and launch sequence rather than adding one copy per bucket.
  return outputs;
}
} // namespace ltm::cuda::fidelity
