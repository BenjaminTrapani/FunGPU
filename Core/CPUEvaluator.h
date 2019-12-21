#include "Compiler.h"
#include "GarbageCollector.h"
#include "PortableMemPool.hpp"
#include "RuntimeBlock.hpp"
#include <SYCL/sycl.hpp>
#include <array>
#include <memory>

namespace FunGPU {
class CPUEvaluator {
public:
  class DependencyTracker;

  using RuntimeBlock_t = RuntimeBlock<DependencyTracker, 8192 * 4>;
  using GarbageCollector_t = RuntimeBlock_t::GarbageCollector_t;

  class DependencyTracker {
    friend class CPUEvaluator;

  public:
    DependencyTracker() : m_activeBlockCountData(0) {}

    CPUEvaluator::RuntimeBlock_t::Error
    AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);

    Index_t GetActiveBlockCount() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      return activeBlockCount.load();
    }

    void ResetActiveBlockCount() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      activeBlockCount.store(0);
    }

    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    GetBlockAtIndex(const Index_t index) {
      return m_newActiveBlocks[index];
    }

  private:
    std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 4096>
        m_newActiveBlocks;
    Index_t m_activeBlockCountData;
  };

  CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool);
  ~CPUEvaluator();
  RuntimeBlock_t::RuntimeValue
  EvaluateProgram(const Compiler::ASTNodeHandle &rootNode,
                  Index_t &maxConcurrentBlocksDuringExec);
  cl::sycl::buffer<PortableMemPool> GetMemPoolBuffer() const {
    return m_memPoolBuff;
  }

private:
  void CreateFirstBlock(const Compiler::ASTNodeHandle rootNode);

  std::shared_ptr<DependencyTracker> m_dependencyTracker;
  std::shared_ptr<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>> m_resultValue;
  cl::sycl::buffer<PortableMemPool::Handle<GarbageCollector_t>>
      m_garbageCollectorHandleBuff;
  cl::sycl::buffer<PortableMemPool> m_memPoolBuff;
  cl::sycl::buffer<DependencyTracker> m_dependencyTrackerBuff;
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>> m_resultValueBuff;
  
  cl::sycl::queue m_workQueue;
};
} // namespace FunGPU