#include "Core/Compiler.hpp"
#include "Core/Error.hpp"
#include "Core/GarbageCollector.hpp"
#include "Core/PortableMemPool.hpp"
#include "Core/RuntimeBlock.hpp"
#include <CL/sycl.hpp>
#include <array>
#include <memory>

namespace FunGPU {
class CPUEvaluator {
public:
  class DependencyTracker;

  using RuntimeBlock_t = RuntimeBlock<DependencyTracker, 8192 * 60>;
  using GarbageCollector_t = RuntimeBlock_t::GarbageCollector_t;

  class DependencyTracker {
    friend class CPUEvaluator;

  public:
    Error
    AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);
    void InsertActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &,
                           Index_t);

    Index_t GetActiveBlockCount() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      return activeBlockCount.fetch_add(0);
    }

    void FlipActiveBlocksBuffer(const Index_t newActiveBlockCount = 0) {
      m_activeBlockCountData = newActiveBlockCount;
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
    std::array<
        std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 8192 * 16>, 2>
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
  cl::sycl::buffer<Error> m_errorsPerBlock;
  cl::sycl::buffer<Index_t> m_blockErrorIdx;
  cl::sycl::buffer<bool> m_markingsExpanded;
  cl::sycl::buffer<Index_t> m_managedAllocdCount;
  cl::sycl::buffer<bool> m_requiresGarbageCollection;
  cl::sycl::buffer<Index_t> m_numActiveBlocksBuff;
  cl::sycl::buffer<RuntimeBlock_t::RuntimeValue> m_resultBufferOnHost;
  cl::sycl::queue m_workQueue;
};
} // namespace FunGPU
