#include "Compiler.h"
#include "Error.hpp"
#include "PortableMemPool.hpp"
#include "RuntimeBlock.hpp"
#include <CL/sycl.hpp>
#include <array>
#include <memory>

namespace FunGPU {
class CPUEvaluator {
public:
  class DependencyTracker;

  using RuntimeBlock_t = RuntimeBlock<DependencyTracker>;

  class DependencyTracker {
    friend class CPUEvaluator;

  public:
    Error
    AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);
    Error
    MarkForDeletion(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);

    Index_t GetActiveBlockCount() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      return activeBlockCount.load();
    }

    Index_t GetMarkedForDeletionCount() {
      cl::sycl::atomic<Index_t> markedForDeletionCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_markedForDeletionCountData)));
      return markedForDeletionCount.load();
    }

    void ResetActiveBlockCount() {
      cl::sycl::atomic<Index_t> activeBlockCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_activeBlockCountData)));
      activeBlockCount.store(0);
    }

    void ResetMarkedForDeletionCount() {
      cl::sycl::atomic<Index_t> markedForDeletionCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_markedForDeletionCountData)));
      markedForDeletionCount.store(0);
    }

    void FlipActiveBlocksBuff() {
      m_activeBlocksBufferIdx = (m_activeBlocksBufferIdx + 1) % m_activeBlocks.size();
      m_prevActiveBlocksBufferIdx = (m_prevActiveBlocksBufferIdx + 1) % m_activeBlocks.size();
      ResetActiveBlockCount();
    }

    void FlipMarkedForDeletionBuff() {
      m_markedForDeletionBufferIdx = (m_markedForDeletionBufferIdx + 1) % m_markedForDeletion.size();
      m_prevMarkedForDeletionBufferIdx = (m_prevMarkedForDeletionBufferIdx + 1) % m_markedForDeletion.size();
      ResetMarkedForDeletionCount();
    }

    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    GetBlockAtIndex(const Index_t index) {
      return m_activeBlocks[m_prevActiveBlocksBufferIdx][index];
    }

    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    GetDeletionAtIndex(const Index_t index) {
      return m_markedForDeletion[m_prevMarkedForDeletionBufferIdx][index];
    }

  private:
    std::array<std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 1024>, 2>
        m_activeBlocks;
    std::array<std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 1024>, 2> m_markedForDeletion;
    Index_t m_activeBlockCountData = 0;
    Index_t m_activeBlocksBufferIdx = 0;
    Index_t m_prevActiveBlocksBufferIdx = 1;
    Index_t m_markedForDeletionCountData = 0;
    Index_t m_markedForDeletionBufferIdx = 0;
    Index_t m_prevMarkedForDeletionBufferIdx = 1;
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
  void DeallocBlocksMarkedForDeletion();

  std::shared_ptr<DependencyTracker> m_dependencyTracker;
  std::shared_ptr<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>
      m_resultValue;

  cl::sycl::buffer<PortableMemPool> m_memPoolBuff;
  cl::sycl::buffer<DependencyTracker> m_dependencyTrackerBuff;
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>
      m_resultValueBuff;
  cl::sycl::buffer<Error> m_errorsPerBlock;
  cl::sycl::buffer<Index_t> m_blockErrorIdx;
  cl::sycl::buffer<bool> m_markingsExpanded;
  cl::sycl::buffer<Index_t> m_numActiveBlocksBuff;
  cl::sycl::buffer<Index_t> m_blocksMarkedForDeletionCount;
  cl::sycl::buffer<RuntimeBlock_t::RuntimeValue> m_resultBufferOnHost;

  cl::sycl::queue m_workQueue;
};
} // namespace FunGPU
