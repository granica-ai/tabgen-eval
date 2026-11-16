#pragma once
#include <algorithm>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <omp.h>

namespace ltm {
// OpenMP ICVs and MKL's local setting belong to the calling thread. Restore
// them on every exit; do not change PyTorch's process-wide thread pool or
// precision.
/** @brief RAII budget for calling-thread OpenMP/MKL settings.
 * @param n Positive requested local numerical worker count (clamped to one).
 * @note Restores prior local settings on every exit; does not mutate PyTorch's
 * process-wide thread/precision settings. Copying this guard is unsupported.
 */
struct ThreadBudget {
  int previous;
  using Setter = int (*)(int);
  Setter setter = nullptr;
  int mkl_previous = 0;
  ThreadBudget(const ThreadBudget&) = delete;
  ThreadBudget& operator=(const ThreadBudget&) = delete;
  explicit ThreadBudget(int n) : previous(omp_get_max_threads()) {
    omp_set_num_threads(std::max(n, 1));
#if defined(__linux__)
    void* handle = dlopen("libtorch_cpu.so", RTLD_LAZY | RTLD_NOLOAD);
    if (handle) {
      // The lowercase symbol is MKL's Fortran entry point (int* argument).
      // Resolve the C entry point, which accepts the thread count by value.
      setter = reinterpret_cast<Setter>(dlsym(handle, "MKL_Set_Num_Threads_Local"));
      if (setter)
        mkl_previous = setter(std::max(n, 1));
      dlclose(handle);
    }
#endif
  }
  ~ThreadBudget() {
    if (setter)
      setter(mkl_previous);
    omp_set_num_threads(previous);
  }
};

} // namespace ltm
