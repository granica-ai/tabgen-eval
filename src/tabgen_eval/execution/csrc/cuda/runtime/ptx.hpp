#pragma once

#include "cuda/runtime/errors.hpp"
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

namespace ltm::cuda::runtime {
// CUDA modules intentionally live until process exit. Unloading from a static
// destructor can run after CUDA context teardown or while launches are pending.
// Function groups are immutable after publication, so launch needs no cache lock.
class PtxModule {
public:
  explicit PtxModule(const std::string& path, const char* missing, const char* context) {
    std::ifstream source(path, std::ios::binary);
    TORCH_CHECK(source, missing, path);
    std::string ptx((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    check_driver(cuModuleLoadData(&module_, ptx.c_str()), context);
  }
  CUfunction function(const char* name, const char* context) const {
    CUfunction result;
    check_driver(cuModuleGetFunction(&result, module_, name), context);
    return result;
  }

private:
  CUmodule module_;
};
class KernelGroup {
public:
  KernelGroup(std::shared_ptr<PtxModule> module, std::initializer_list<const char*> names,
              const char* context)
      : module_(std::move(module)), context_(context) {
    for (const auto* name : names)
      functions_.emplace(name, module_->function(name, context_));
  }
  CUfunction function(const char* name) const {
    return functions_.at(name);
  }
  template <class... Args>
  void launch(const char* name, int blocks, int threads, int shared, cudaStream_t stream,
              Args... values) const {
    if (!blocks)
      return;
    void* arguments[] = {const_cast<void*>(static_cast<const void*>(&values))...};
    check_driver(cuLaunchKernel(function(name), blocks, 1, 1, threads, 1, 1, shared,
                                reinterpret_cast<CUstream>(stream), arguments, nullptr),
                 context_);
  }

private:
  std::shared_ptr<PtxModule> module_;
  std::map<std::string, CUfunction> functions_;
  const char* context_;
};
inline std::shared_ptr<KernelGroup> kernels(const std::string& path, const char* group,
                                            std::initializer_list<const char*> names,
                                            const char* missing, const char* context) {
  struct Cache {
    std::mutex mutex;
    std::map<std::pair<int, std::string>, std::shared_ptr<PtxModule>> modules;
    std::map<std::tuple<int, std::string, std::string>, std::shared_ptr<KernelGroup>> groups;
  };
  static auto* cache = new Cache;
  int device = 0;
  check_runtime(cudaGetDevice(&device), "native PTX CUDA error: ");
  std::lock_guard<std::mutex> lock(cache->mutex);
  auto group_key = std::make_tuple(device, path, std::string(group));
  auto found = cache->groups.find(group_key);
  if (found != cache->groups.end())
    return found->second;
  auto module_key = std::make_pair(device, path);
  auto& module = cache->modules[module_key];
  if (!module)
    module = std::make_shared<PtxModule>(path, missing, context);
  auto result = std::make_shared<KernelGroup>(module, names, context);
  cache->groups.emplace(group_key, result);
  return result;
}
} // namespace ltm::cuda::runtime
