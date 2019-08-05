#include "Compiler.h"
#include "PortableMemPool.hpp"
#include "RuntimeBlock.hpp"
#include <SYCL/sycl.hpp>
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
    DependencyTracker() : m_activeBlockCount(0), m_removeRefBlocksCount(0) {}
    void AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block,
                        const PortableMemPool::DeviceAccessor_t &memPoolAcc);
    void MarkRequiresRemoveRef(
        const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);

    unsigned int GetActiveBlockCount() { return m_activeBlockCount.load(); }
    unsigned int GetRequiresRemoveRefCount() {
      return m_removeRefBlocksCount.load();
    }
    void ResetActiveBlockCount() { m_activeBlockCount.store(0); }
    void ResetRequiresRemoveRefCount() { m_removeRefBlocksCount.store(0); }

    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    GetBlockAtIndex(const unsigned int index) {
      return m_newActiveBlocks[index];
    }
    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    GetRemoveRefBlockAtIndex(const unsigned int index) {
      return m_requireRemoveRefBlocks[index];
    }

  private:
    std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 4096>
        m_newActiveBlocks;
    std::atomic<unsigned int> m_activeBlockCount;

    std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 4096>
        m_requireRemoveRefBlocks;
    std::atomic<unsigned int> m_removeRefBlocksCount;
  };

  CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool);
  ~CPUEvaluator();
  RuntimeBlock_t::RuntimeValue
  EvaluateProgram(const Compiler::ASTNodeHandle &rootNode,
                  unsigned int &maxConcurrentBlocksDuringExec);
  cl::sycl::buffer<PortableMemPool> GetMemPoolBuffer() const {
    return m_memPoolBuff;
  }

private:
  void CreateFirstBlock(const Compiler::ASTNodeHandle rootNode);

  PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue> m_resultValue;
  std::shared_ptr<DependencyTracker> m_dependencyTracker;
  cl::sycl::buffer<PortableMemPool> m_memPoolBuff;
  cl::sycl::buffer<DependencyTracker> m_dependencyTrackerBuff;

  cl::sycl::queue m_workQueue;
};
}