#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <climits>
#include <cstdint>
#include <cuda.h>
#include <dlfcn.h>

namespace ltm::cuda::runtime {
// Triton/Python owns the CUfunction and its module lifetime. This view validates
// launch metadata and borrows the function without loading or unloading it.
struct Driver {
  decltype(&cuLaunchKernel) launch;
  Driver() {
    // Keep the driver loaded for the process lifetime, like Triton's launcher.
    void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(handle, "Conditional cannot load libcuda.so.1");
    launch = reinterpret_cast<decltype(launch)>(dlsym(handle, "cuLaunchKernel"));
    TORCH_CHECK(launch, "Conditional cannot resolve cuLaunchKernel");
  }
};
class BorrowedKernel {
public:
  CUfunction function;
  unsigned threads, shared;
  explicit BorrowedKernel(at::IntArrayRef spec) {
    TORCH_CHECK(spec.size() == 3 && spec[0] != 0 && spec[1] > 0 && spec[1] <= 32 && spec[2] >= 0 &&
                    spec[2] <= UINT_MAX,
                "invalid Conditional kernel launch metadata");
    function = reinterpret_cast<CUfunction>(static_cast<uintptr_t>(spec[0]));
    threads = 32 * spec[1];
    shared = spec[2];
  }
  void run(unsigned x, unsigned y, unsigned z, CUstream stream, void** args) const {
    if (!x || !y || !z)
      return;
    static Driver driver;
    auto status = driver.launch(function, x, y, z, threads, 1, 1, shared, stream, args, nullptr);
    TORCH_CHECK(status == CUDA_SUCCESS, "Conditional cuLaunchKernel error: ", int(status));
  }
};

} // namespace ltm::cuda::runtime
