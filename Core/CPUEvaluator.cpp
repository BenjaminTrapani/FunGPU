#include "CPUEvaluator.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cl::sycl;

namespace FunGPU {
class create_gc;
class init_first_block;
class run_eval_pass;
class prep_blocks;
class check_requires_garbage_collection;
class get_managed_allocd_count;
class gc_initial_mark;
class gc_mark;
class gc_sweep;
class gc_prepare_compact;
class gc_compact;

CPUEvaluator::RuntimeBlock_t::Error
CPUEvaluator::DependencyTracker::AddActiveBlock(
    const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block) {
  cl::sycl::atomic<Index_t> activeBlockCount(
      (cl::sycl::multi_ptr<Index_t,
                           cl::sycl::access::address_space::global_space>(
          &m_activeBlockCountData)));
  const auto indexToInsert = activeBlockCount.fetch_add(1);
  if (indexToInsert >= m_newActiveBlocks.size()) {
    return RuntimeBlock_t::Error(RuntimeBlock_t::Error::Type::OutOfMemory,
                                 "Ran out of space for active blocks");
  }
  m_newActiveBlocks[indexToInsert] = block;

  return RuntimeBlock_t::Error();
}

CPUEvaluator::CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool)
    : m_dependencyTracker(std::make_shared<DependencyTracker>()),
      m_resultValue(std::make_shared<
                    PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>()),
      m_garbageCollectorHandleBuff(range<1>(1)), m_memPoolBuff(memPool),
      m_dependencyTrackerBuff(m_dependencyTracker, range<1>(1)),
      m_resultValueBuff(m_resultValue, range<1>(1))
/* m_workQueue(host_selector{})*/ {
  std::cout << std::endl;
  std::cout << "Running on "
            << m_workQueue.get_device().get_info<info::device::name>()
            << std::endl;

  try {
    m_workQueue.submit([&](handler &cgh) {
      auto memPoolWrite =
          m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      auto garbageCollectorHandleAcc =
          m_garbageCollectorHandleBuff.get_access<access::mode::read_write>(
              cgh);
      auto memPoolHostAcc =
          m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      auto resultValueWrite =
          m_resultValueBuff.get_access<access::mode::discard_write>(cgh);
      cgh.single_task<class create_gc>(
          [memPoolWrite, garbageCollectorHandleAcc, resultValueWrite]() {
            garbageCollectorHandleAcc[0] =
                memPoolWrite[0].Alloc<GarbageCollector_t>(memPoolWrite);
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
    auto gcHandleAcc =
        m_garbageCollectorHandleBuff.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class evaluator_dealloc>(
        [memPoolAcc, gcHandleAcc, resultValueAcc]() {
          memPoolAcc[0].Dealloc(resultValueAcc[0]);
          memPoolAcc[0].Dealloc(gcHandleAcc[0]);
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
      auto garbageCollectorHandleAcc =
          m_garbageCollectorHandleBuff.get_access<access::mode::read>(cgh);
      auto resultValueAcc =
          m_resultValueBuff.get_access<access::mode::read>(cgh);
      cgh.single_task<class init_first_block>([memPoolWrite, dependencyTracker,
                                               resultValueAcc, rootNode,
                                               garbageCollectorHandleAcc]() {
        auto gcRef = memPoolWrite[0].derefHandle(garbageCollectorHandleAcc[0]);
        gcRef->SetMemPoolAcc(memPoolWrite);
        const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
        const auto sharedInitialBlock = gcRef->AllocManaged(
            rootNode, emptyBlock, emptyBlock, dependencyTracker,
            resultValueAcc[0], memPoolWrite, garbageCollectorHandleAcc[0]);
        dependencyTracker[0].AddActiveBlock(sharedInitialBlock);
      });
    });
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }
  std::cout << "Successfully created first block" << std::endl;
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
CPUEvaluator::EvaluateProgram(const Compiler::ASTNodeHandle &rootNode,
                              Index_t &maxConcurrentBlocksDuringExec) {
  CreateFirstBlock(rootNode);

  maxConcurrentBlocksDuringExec = 0;
  try {
    buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> workingBlocksBuff(
        range<1>(m_dependencyTracker->m_newActiveBlocks.size()));

    buffer<RuntimeBlock_t::Error> errorsPerBlock(
        range<1>(m_dependencyTracker->m_newActiveBlocks.size()));

    Index_t blockErrorIdxData = 0;
    buffer<Index_t> blockErrorIdx(&blockErrorIdxData, range<1>(1));

    bool markingsExpandedData = false;
    buffer<bool> markingsExpanded(&markingsExpandedData, range<1>(1));

    Index_t managedAllocdCountData = 0;
    buffer<Index_t> managedAllocdCount(&managedAllocdCountData, range<1>(1));

    Index_t runtimeValuesRequiredCountData = 0;
    buffer<Index_t> runtimeValuesRequiredCount(&runtimeValuesRequiredCountData,
                                               range<1>(1));
    Index_t runtimeBlocksRequiredCountData = 0;
    buffer<Index_t> runtimeBlocksRequiredCount(&runtimeBlocksRequiredCountData,
                                               range<1>(1));
    bool requiresGarbageCollectionData = false;
    buffer<bool> requiresGarbageCollection(&requiresGarbageCollectionData,
                                           range<1>(1));
    Index_t numActiveBlocksData = 0;
    buffer<Index_t> numActiveBlocksBuff(&numActiveBlocksData, range<1>(1));
    while (true) {
      {
        m_workQueue.submit([&](handler &cgh) {
          auto numActiveBlocksWrite =
              numActiveBlocksBuff.get_access<access::mode::write>(cgh);
          auto depTrackerAcc =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
          cgh.single_task<class update_active_block_count>(
              [numActiveBlocksWrite, depTrackerAcc]() {
                numActiveBlocksWrite[0] =
                    depTrackerAcc[0].GetActiveBlockCount();
                depTrackerAcc[0].ResetActiveBlockCount();
              });
        });

        auto runtimeValuesRequiredAcc =
            runtimeValuesRequiredCount.get_access<access::mode::write>();
        runtimeValuesRequiredAcc[0] = 0;
        auto runtimeBlocksRequiredAcc =
            runtimeBlocksRequiredCount.get_access<access::mode::write>();
        runtimeBlocksRequiredAcc[0] = 0;

        auto hostBlockHasErrorAcc =
            blockErrorIdx.get_access<access::mode::read_write>();
        if (hostBlockHasErrorAcc[0] > 0) {
          auto errorsPerBlockAcc =
              errorsPerBlock.get_access<access::mode::read_write>();
          std::stringstream ss;
          ss << "Block errors in previous pass: " << std::endl;
          for (Index_t i = 0; i < hostBlockHasErrorAcc[0]; ++i) {
            ss << errorsPerBlockAcc[i].GetDescription() << std::endl;
          }
          const auto errorString = ss.str();
          throw std::runtime_error(errorString);
        }
        hostBlockHasErrorAcc[0] = 0;
      }

      const auto numActiveBlocks =
          numActiveBlocksBuff.get_access<access::mode::read>()[0];
      maxConcurrentBlocksDuringExec =
          std::max(numActiveBlocks, maxConcurrentBlocksDuringExec);

      if (numActiveBlocks > 0) {
        m_workQueue.submit([&](handler &cgh) {
          auto dependencyTracker =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
          auto workingBlocksAcc =
              workingBlocksBuff.get_access<access::mode::read_write>(cgh);
          auto memPoolAcc =
              m_memPoolBuff.get_access<access::mode::read_write>(cgh);

          auto runtimeValuesRequiredCountAtomic =
              runtimeValuesRequiredCount.get_access<access::mode::atomic>(cgh);
          auto runtimeBlocksRequiredCountAtomic =
              runtimeBlocksRequiredCount.get_access<access::mode::atomic>(cgh);
          cgh.parallel_for<class prep_blocks>(
              cl::sycl::range<1>(numActiveBlocks),
              [dependencyTracker, workingBlocksAcc, memPoolAcc,
               runtimeValuesRequiredCountAtomic,
               runtimeBlocksRequiredCountAtomic](item<1> itm) {
                const auto idx = itm.get_linear_id();
                workingBlocksAcc[idx] = dependencyTracker[0].GetBlockAtIndex(
                    static_cast<Index_t>(idx));
                const auto derefdWorking =
                    memPoolAcc[0].derefHandle(workingBlocksAcc[idx]);
                const auto requiredSpace = derefdWorking->GetRequiredAllocs();
                runtimeValuesRequiredCountAtomic[0].fetch_add(
                    requiredSpace.runtimeValuesRequired);
                runtimeBlocksRequiredCountAtomic[0].fetch_add(
                    requiredSpace.dependentBlocksRequired);
              });
        });
      }

      m_workQueue.submit([&](handler &cgh) {
        auto memPoolAcc =
            m_memPoolBuff.get_access<access::mode::read_write>(cgh);
        auto runtimeValuesRequiredCountAcc =
            runtimeValuesRequiredCount.get_access<access::mode::read>(cgh);
        auto runtimeBlocksRequiredCountAcc =
            runtimeBlocksRequiredCount.get_access<access::mode::read>(cgh);
        auto requiresGarbageCollectAcc =
            requiresGarbageCollection.get_access<access::mode::write>(cgh);
        cgh.single_task<class check_requires_garbage_collection>(
            [memPoolAcc, runtimeValuesRequiredCountAcc,
             runtimeBlocksRequiredCountAcc, requiresGarbageCollectAcc] {
              const auto numRuntimeValuesFree =
                  memPoolAcc[0].GetNumFree<RuntimeBlock_t::RuntimeValue>();
              const auto numRuntimeBlocksFree =
                  memPoolAcc[0].GetNumFree<RuntimeBlock_t>();
              requiresGarbageCollectAcc[0] =
                  numRuntimeValuesFree < runtimeValuesRequiredCountAcc[0] ||
                  numRuntimeBlocksFree < runtimeBlocksRequiredCountAcc[0];
            });
      });

      bool requiresGarbageCollectOnHost;
      {
        auto requiresGarbageCollectAcc =
            requiresGarbageCollection.get_access<access::mode::read>();
        requiresGarbageCollectOnHost = requiresGarbageCollectAcc[0];
      }

      if (requiresGarbageCollectOnHost || numActiveBlocks == 0) {
        if (numActiveBlocks > 0) {
          m_workQueue.submit([&](handler &cgh) {
            auto workingBlocksAcc =
                workingBlocksBuff.get_access<access::mode::read_write>(cgh);
            auto memPoolAcc =
                m_memPoolBuff.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class gc_initial_mark>(
                cl::sycl::range<1>(numActiveBlocks),
                [workingBlocksAcc, memPoolAcc](item<1> itm) {
                  const auto idx = itm.get_linear_id();
                  const auto derefdWorking =
                      memPoolAcc[0].derefHandle(workingBlocksAcc[idx]);
                  derefdWorking->SetMarked();
                });
          });
        }

        m_workQueue.submit([&](handler &cgh) {
          auto managedAllocdCountDevice =
              managedAllocdCount.get_access<access::mode::read_write>(cgh);
          auto memPoolAcc =
              m_memPoolBuff.get_access<access::mode::read_write>(cgh);
          auto gcHandleAcc =
              m_garbageCollectorHandleBuff.get_access<access::mode::read>(cgh);
          cgh.single_task<class get_managed_allocd_count>(
              [managedAllocdCountDevice, memPoolAcc, gcHandleAcc] {
                auto gcRef = memPoolAcc[0].derefHandle(gcHandleAcc[0]);
                managedAllocdCountDevice[0] =
                    gcRef->GetManagedAllocationCount();
              });
        });

        Index_t managedAllocdSize;
        {
          auto managedAllocdCountHost =
              managedAllocdCount.get_access<access::mode::read>();
          managedAllocdSize = managedAllocdCountHost[0];
        }
        if (managedAllocdSize > 0) {
          // Expand markings
          bool wasMarkingsExpandedInLastPass = true;
          while (wasMarkingsExpandedInLastPass) {
            m_workQueue.submit([&](handler &cgh) {
              auto gcHandleAcc =
                  m_garbageCollectorHandleBuff.get_access<access::mode::read>(
                      cgh);
              auto memPoolAcc =
                  m_memPoolBuff.get_access<access::mode::read_write>(cgh);
              auto markingsExpandedAcc =
                  markingsExpanded.get_access<access::mode::read_write>(cgh);
              cgh.parallel_for<class gc_mark>(
                  cl::sycl::range<1>(managedAllocdSize),
                  [gcHandleAcc, memPoolAcc, markingsExpandedAcc](item<1> itm) {
                    auto gcRef = memPoolAcc[0].derefHandle(gcHandleAcc[0]);
                    gcRef->SetMemPoolAcc(memPoolAcc);
                    const auto wereMarkingsExpandedHere = gcRef->RunMarkPass(
                        static_cast<Index_t>(itm.get_linear_id()));
                    if (wereMarkingsExpandedHere) {
                      markingsExpandedAcc[0] = true;
                    }
                  });
            });
            {
              auto markingsExpandedHostAcc =
                  markingsExpanded.get_access<access::mode::read_write>();
              wasMarkingsExpandedInLastPass = markingsExpandedHostAcc[0];
              markingsExpandedHostAcc[0] = false;
            }
          }

          // Sweep unmarked, one pass
          m_workQueue.submit([&](handler &cgh) {
            auto gcHandleAcc =
                m_garbageCollectorHandleBuff.get_access<access::mode::read>(
                    cgh);
            auto memPoolAcc =
                m_memPoolBuff.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class gc_sweep>(
                cl::sycl::range<1>(managedAllocdSize),
                [gcHandleAcc, memPoolAcc](item<1> itm) {
                  auto gcRef = memPoolAcc[0].derefHandle(gcHandleAcc[0]);
                  gcRef->SetMemPoolAcc(memPoolAcc);
                  gcRef->Sweep(static_cast<Index_t>(itm.get_linear_id()));
                });
          });

          m_workQueue.submit([&](handler &cgh) {
            auto gcHandleAcc =
                m_garbageCollectorHandleBuff.get_access<access::mode::read>(
                    cgh);
            auto memPoolAcc =
                m_memPoolBuff.get_access<access::mode::read_write>(cgh);
            cgh.single_task<class gc_prepare_compact>(
                [gcHandleAcc, memPoolAcc]() {
                  auto gcRef = memPoolAcc[0].derefHandle(gcHandleAcc[0]);
                  gcRef->ResetAllocationCount();
                });
          });

          // Compact
          m_workQueue.submit([&](handler &cgh) {
            auto gcHandleAcc =
                m_garbageCollectorHandleBuff.get_access<access::mode::read>(
                    cgh);
            auto memPoolAcc =
                m_memPoolBuff.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class gc_compact>(
                cl::sycl::range<1>(managedAllocdSize),
                [gcHandleAcc, memPoolAcc](item<1> itm) {
                  auto gcRef = memPoolAcc[0].derefHandle(gcHandleAcc[0]);
                  gcRef->Compact(static_cast<Index_t>(itm.get_linear_id()));
                });
          });
        }
      }

      if (numActiveBlocks > 0) {
        // main eval pass
        m_workQueue.submit([&](handler &cgh) {
          auto memPoolWrite =
              m_memPoolBuff.get_access<access::mode::read_write>(cgh);
          auto dependencyTracker =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
          auto gcHandleAcc =
              m_garbageCollectorHandleBuff.get_access<access::mode::read>(cgh);
          auto workingBlocksAcc =
              workingBlocksBuff.get_access<access::mode::read_write>(cgh);

          auto blockErrorIdxAtomicAcc =
              blockErrorIdx.get_access<access::mode::atomic>(cgh);
          auto blockErrorAcc =
              errorsPerBlock.get_access<access::mode::read_write>(cgh);

          cgh.parallel_for<class run_eval_pass>(
              cl::sycl::range<1>(numActiveBlocks),
              [dependencyTracker, memPoolWrite, workingBlocksAcc,
               blockErrorIdxAtomicAcc, blockErrorAcc,
               gcHandleAcc](item<1> itm) {
                auto currentBlock = workingBlocksAcc[itm.get_linear_id()];
                auto derefdCurrentBlock =
                    memPoolWrite[0].derefHandle(currentBlock);
                auto gcRef = memPoolWrite[0].derefHandle(gcHandleAcc[0]);
                gcRef->SetMemPoolAcc(memPoolWrite);
                derefdCurrentBlock->SetResources(memPoolWrite,
                                                 dependencyTracker);
                const auto error = derefdCurrentBlock->PerformEvalPass();
                if (error.GetType() != RuntimeBlock_t::Error::Type::Success) {
                  cl::sycl::atomic<Index_t> blockErrorIdxAtomic(
                      blockErrorIdxAtomicAcc[0]);
                  blockErrorAcc[blockErrorIdxAtomic.fetch_add(1)] = error;
                }
              });
        });
      } else {
        break;
      }
    }
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }

  RuntimeBlock_t::RuntimeValue resultOnHostData;
  buffer<RuntimeBlock_t::RuntimeValue> resultBufferOnHost(&resultOnHostData,
                                                          range<1>(1));
  m_workQueue.submit([&](handler &cgh) {
    auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
    auto resultOnHostAcc =
        resultBufferOnHost.get_access<access::mode::discard_write>(cgh);
    auto resultValRefAcc =
        m_resultValueBuff.get_access<access::mode::read>(cgh);
    cgh.single_task<class fetch_result>(
        [resultOnHostAcc, memPoolAcc, resultValRefAcc]() {
          resultOnHostAcc[0] = *memPoolAcc[0].derefHandle(resultValRefAcc[0]);
        });
  });
  {
    auto resultOnHostAcc = resultBufferOnHost.get_access<access::mode::read>();
    return resultOnHostAcc[0];
  }
}
} // namespace FunGPU
