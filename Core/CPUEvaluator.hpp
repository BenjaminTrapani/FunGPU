#include "Compiler.hpp"
#include "Error.hpp"
#include "GarbageCollector.hpp"
#include "PortableMemPool.hpp"
#include "RuntimeBlock.hpp"
#include <CL/sycl.hpp>
#include <array>
#include <memory>

namespace FunGPU {
class CPUEvaluator {
public:
  class DependencyTracker;

  using RuntimeBlock_t = RuntimeBlock<DependencyTracker, 8192 * 52>;
  using GarbageCollector_t = RuntimeBlock_t::GarbageCollector_t;

  class DependencyTracker {
    friend class CPUEvaluator;

  public:
    Error
    AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);

    Index_t GetActiveBlockCount() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      return activeBlockCount.load();
    }

    void FlipActiveBlocksBuffer() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      activeBlockCount.store(0);
      m_activeBlocksBufferIdx =
          (m_activeBlocksBufferIdx + 1) % m_activeBlocks.size();
      m_prevActiveBlocksBufferIdx =
          (m_prevActiveBlocksBufferIdx + 1) % m_activeBlocks.size();
    }

    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    GetBlockAtIndex(const Index_t index) {
      return m_activeBlocks[m_prevActiveBlocksBufferIdx][index];
    }

  private:
    std::array<std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 8192 * 8>,
               2>
        m_activeBlocks;
    Index_t m_activeBlockCountData = 0;
    Index_t m_activeBlocksBufferIdx = 0;
    Index_t m_prevActiveBlocksBufferIdx = 1;
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
  void CreateFirstBlock(Compiler::ASTNodeHandle rootNode);
  bool ComputeRequiredResourcesForActiveSet(Index_t numActiveBlocks);
  void CheckForBlockErrors(Index_t maxConcurrentBlocksDuringExec);
  void PerformGarbageCollection(Index_t numActiveBlocks);

  std::shared_ptr<DependencyTracker> m_dependencyTracker;
  std::shared_ptr<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>
      m_resultValue;

  cl::sycl::buffer<PortableMemPool::Handle<GarbageCollector_t>>
      m_garbageCollectorHandleBuff;
  cl::sycl::buffer<PortableMemPool> m_memPoolBuff;
  cl::sycl::buffer<DependencyTracker> m_dependencyTrackerBuff;
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>
      m_resultValueBuff;
  cl::sycl::buffer<Error> m_errorsPerBlock;  cl::sycl::buffer<Index_t> m_blockErrorIdx;
  cl::sycl::buffer<bool> m_markingsExpanded;
  cl::sycl::buffer<Index_t> m_managedAllocdCount;
  cl::sycl::buffer<bool> m_requiresGarbageCollection;
  cl::sycl::buffer<Index_t> m_numActiveBlocksBuff;
  cl::sycl::buffer<RuntimeBlock_t::RuntimeValue> m_resultBufferOnHost;
  cl::sycl::buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> m_notReservedBlocksBuff;
  cl::sycl::buffer<Index_t> m_notReservedBlocksCount;
  cl::sycl::queue m_workQueue;
};
} // namespace FunGPU
