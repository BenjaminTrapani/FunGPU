#include "CPUEvaluator.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cl::sycl;

namespace FunGPU {
class create_result_value;
class evaluator_dealloc;
  class prep_blocks;
  class fetch_num_blocks_to_delete;
  class dealloc_deleted_blocks;
  class init_first_block;
  class update_active_block_count;
  class run_eval_pass;
  class fetch_result;

  Error CPUEvaluator::DependencyTracker::AddActiveBlock(
                     const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block) {
    cl::sycl::atomic<Index_t> activeBlockCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_activeBlockCountData)));
    const auto indexToInsert = activeBlockCount.fetch_add(1);
    auto& destArray = m_activeBlocks[m_activeBlocksBufferIdx];
    if (indexToInsert >= destArray.size()) {
      return Error(Error::Type::EvaluatorOutOfActiveBlocks);
    }
    destArray[indexToInsert] = block;

    return Error();
  }

  Error CPUEvaluator::DependencyTracker::MarkForDeletion(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block) {
    cl::sycl::atomic<Index_t> markedForDeletionCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_markedForDeletionCountData)));
    const auto indexToInsert = markedForDeletionCount.fetch_add(1);
    auto& destArray = m_markedForDeletion[m_markedForDeletionBufferIdx]; 
    if (indexToInsert >= destArray.size()) {
      return Error(Error::Type::EvaluatorOutOfDeletionBlocks);
    }
    destArray[indexToInsert] = block;

    return Error();
  }

CPUEvaluator::CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool)
    : m_dependencyTracker(std::make_shared<DependencyTracker>()),
      m_resultValue(std::make_shared<
                    PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>()),
      m_memPoolBuff(memPool),
      m_dependencyTrackerBuff(m_dependencyTracker, range<1>(1)),
      m_resultValueBuff(m_resultValue, range<1>(1)),
      m_errorsPerBlock(range<1>(m_dependencyTracker->m_activeBlocks[0].size())),
      m_blockErrorIdx(range<1>(1)), m_markingsExpanded(range<1>(1)),
      m_numActiveBlocksBuff(range<1>(1)),
      m_blocksMarkedForDeletionCount(range<1>(1)),
      m_resultBufferOnHost(range<1>(1))
/* m_workQueue(host_selector{})*/ {
  std::cout << std::endl;
  std::cout << "Running on "
            << m_workQueue.get_device().get_info<info::device::name>()
            << std::endl;
  std::cout << "Runtime block size in bytes: " << sizeof(RuntimeBlock_t)
            << std::endl;
  std::cout << "Runtime value size in bytes: "
            << sizeof(typename RuntimeBlock_t::RuntimeValue) << std::endl;
  try {
  m_workQueue.submit([&](handler &cgh) {
      auto memPoolWrite =
          m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      auto resultValueWrite =
          m_resultValueBuff.get_access<access::mode::discard_write>(cgh);
      cgh.single_task<class create_result_value>(
                                       [memPoolWrite, resultValueWrite]() {
            resultValueWrite[0] =
                       memPoolWrite[0].Alloc<RuntimeBlock_t::RuntimeValue>();
          });
    });
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception in init: " << e.what() << std::endl;
  }
}

CPUEvaluator::~CPUEvaluator() {
  m_workQueue.submit([&](handler &cgh) {
    auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
    auto resultValueAcc =
        m_resultValueBuff.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class evaluator_dealloc>(
        [memPoolAcc, resultValueAcc]() {
          memPoolAcc[0].Dealloc(resultValueAcc[0]);
        });
  });
}

void CPUEvaluator::CreateFirstBlock(const Compiler::ASTNodeHandle rootNode) {
  try {
    m_workQueue.submit([&](handler &cgh) {
      auto memPoolWrite =
          m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      auto dependencyTracker =
          m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
      auto resultValueAcc =
          m_resultValueBuff.get_access<access::mode::read>(cgh);
      cgh.single_task<class init_first_block>(
          [memPoolWrite, dependencyTracker, resultValueAcc, rootNode]() {
            const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
            const auto sharedInitialBlock =
              memPoolWrite[0].template Alloc<RuntimeBlock_t>(
                        rootNode, emptyBlock, emptyBlock, dependencyTracker,
                        resultValueAcc[0], memPoolWrite);
            dependencyTracker[0].AddActiveBlock(sharedInitialBlock);
            auto derefdInitialBlock = memPoolWrite[0].derefHandle(sharedInitialBlock);
          });
    });
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }
  std::cout << "Successfully created first block" << std::endl;
}

void CPUEvaluator::DeallocBlocksMarkedForDeletion() {
  m_workQueue.wait();
  while (true) {
  m_workQueue.submit([&](handler& cgh){
                       auto dependencyTrackerAcc = m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
                       auto numBlocksMarkedForDeletionAcc = m_blocksMarkedForDeletionCount.get_access<access::mode::discard_write>(cgh);
                       cgh.single_task<class fetch_num_blocks_to_delete>([dependencyTrackerAcc, numBlocksMarkedForDeletionAcc] () {
                                                                           numBlocksMarkedForDeletionAcc[0] = dependencyTrackerAcc[0].GetMarkedForDeletionCount();
                                                                           dependencyTrackerAcc[0].FlipMarkedForDeletionBuff();
                                                                   });
                     });

  const auto numBlocksMarkedForDeletion = m_blocksMarkedForDeletionCount.get_access<access::mode::read>()[0];
  if (numBlocksMarkedForDeletion > 0) {
    m_workQueue.submit([&](handler &cgh) {
      auto dependencyTrackerAcc =
          m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
      auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class dealloc_deleted_blocks>(
          range<1>(numBlocksMarkedForDeletion),
          [dependencyTrackerAcc, memPoolAcc](const item<1> itm) {
            const auto handleToDelete =  dependencyTrackerAcc[0].GetDeletionAtIndex(itm.get_linear_id());
            auto derefdBlock = memPoolAcc[0].derefHandle(handleToDelete);
            derefdBlock->SetResources(memPoolAcc, dependencyTrackerAcc);
            const auto clearRefsError = derefdBlock->ClearRefs();
            if (clearRefsError.GetType() != Error::Type::Success) {
              // TODO handle the error
            }
            memPoolAcc[0].Dealloc(handleToDelete);
          });
          });
  } else {
    break;
    }
 }
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
CPUEvaluator::EvaluateProgram(const Compiler::ASTNodeHandle &rootNode,
                              Index_t &maxConcurrentBlocksDuringExec) {
  CreateFirstBlock(rootNode);

  maxConcurrentBlocksDuringExec = 0;
  m_blockErrorIdx.get_access<access::mode::discard_write>()[0] = 0;
  try {
    while (true) {
      m_workQueue.submit([&](handler &cgh) {
        auto numActiveBlocksWrite =
            m_numActiveBlocksBuff.get_access<access::mode::discard_write>(cgh);
        auto depTrackerAcc =
            m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class update_active_block_count>(
            [numActiveBlocksWrite, depTrackerAcc]() {
              numActiveBlocksWrite[0] = depTrackerAcc[0].GetActiveBlockCount();
              depTrackerAcc[0].FlipActiveBlocksBuff();
            });
      });
      {
        auto hostBlockHasErrorAcc =
            m_blockErrorIdx.get_access<access::mode::read_write>();
        if (hostBlockHasErrorAcc[0] > 0) {
          auto errorsPerBlockAcc =
              m_errorsPerBlock.get_access<access::mode::read_write>();
          std::stringstream ss;
          ss << "Block errors in previous pass: " << std::endl;
          for (Index_t i = 0; i < hostBlockHasErrorAcc[0]; ++i) {
            ss << static_cast<int>(errorsPerBlockAcc[i].GetType()) << std::endl;
          }
          ss << "Max concurrent blocks during exec: "
             << maxConcurrentBlocksDuringExec << std::endl;
          const auto errorString = ss.str();
          throw std::runtime_error(errorString);
        }
        hostBlockHasErrorAcc[0] = 0;
      }

      const auto numActiveBlocks =
          m_numActiveBlocksBuff.get_access<access::mode::read>()[0];
      maxConcurrentBlocksDuringExec =
          std::max(numActiveBlocks, maxConcurrentBlocksDuringExec);

      if (numActiveBlocks > 0) {
        // main eval pass
        m_workQueue.submit([&](handler &cgh) {
          auto memPoolWrite =
              m_memPoolBuff.get_access<access::mode::read_write>(cgh);
          auto dependencyTracker =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);

          auto blockErrorIdxAtomicAcc =
              m_blockErrorIdx.get_access<access::mode::atomic>(cgh);
          auto blockErrorAcc =
              m_errorsPerBlock.get_access<access::mode::read_write>(cgh);

          cgh.parallel_for<class run_eval_pass>(
              cl::sycl::range<1>(numActiveBlocks),
              [dependencyTracker, memPoolWrite,
               blockErrorIdxAtomicAcc, blockErrorAcc](item<1> itm) {
                auto currentBlock = dependencyTracker[0].GetBlockAtIndex(itm.get_linear_id());
                auto derefdCurrentBlock =
                    memPoolWrite[0].derefHandle(currentBlock);
                derefdCurrentBlock->SetResources(memPoolWrite,
                                                 dependencyTracker);
                const auto error = derefdCurrentBlock->PerformEvalPass();
                if (error.GetType() != Error::Type::Success) {
                  cl::sycl::atomic<Index_t> blockErrorIdxAtomic(
                      blockErrorIdxAtomicAcc[0]);
                  blockErrorAcc[blockErrorIdxAtomic.fetch_add(1)] = error;
                }
              });
        });

        DeallocBlocksMarkedForDeletion();
      } else {
        break;
      }
    }
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }

  m_workQueue.submit([&](handler &cgh) {
    auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
    auto resultOnHostAcc =
        m_resultBufferOnHost.get_access<access::mode::discard_write>(cgh);
    auto resultValRefAcc =
        m_resultValueBuff.get_access<access::mode::read>(cgh);
    cgh.single_task<class fetch_result>(
        [resultOnHostAcc, memPoolAcc, resultValRefAcc]() {
          resultOnHostAcc[0] = *memPoolAcc[0].derefHandle(resultValRefAcc[0]);
        });
  });
  {
    auto resultOnHostAcc =
        m_resultBufferOnHost.get_access<access::mode::read>();
    return resultOnHostAcc[0];
  }
}
} // namespace FunGPU
