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

Error CPUEvaluator::DependencyTracker::AddActiveBlock(
    const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block) {
  cl::sycl::atomic<Index_t> activeBlockCount(
      (cl::sycl::multi_ptr<Index_t,
                           cl::sycl::access::address_space::global_space>(
          &m_activeBlockCountData)));
  const auto indexToInsert = activeBlockCount.fetch_add(1);
  if (indexToInsert >= m_newActiveBlocks.size()) {
    return Error(Error::Type::EvaluatorOutOfActiveBlocks);
  }
  m_newActiveBlocks[indexToInsert] = block;

  return Error();
}

CPUEvaluator::CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool)
    : m_dependencyTracker(std::make_shared<DependencyTracker>()),
      m_resultValue(std::make_shared<
                    PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>()),
      m_garbageCollectorHandleBuff(range<1>(1)), m_memPoolBuff(memPool),
      m_dependencyTrackerBuff(m_dependencyTracker, range<1>(1)),
      m_resultValueBuff(m_resultValue, range<1>(1)),
      m_workingBlocksBuff(
          range<1>(m_dependencyTracker->m_newActiveBlocks.size())),
      m_errorsPerBlock(range<1>(m_dependencyTracker->m_newActiveBlocks.size())),
      m_blockErrorIdx(range<1>(1)), m_markingsExpanded(range<1>(1)),
      m_managedAllocdCount(range<1>(1)),
      m_runtimeValuesRequiredCount(range<1>(1)),
      m_runtimeBlocksRequiredCount(range<1>(1)),
      m_requiresGarbageCollection(range<1>(1)),
      m_numActiveBlocksBuff(range<1>(1)), m_resultBufferOnHost(range<1>(1))
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
    cl::sycl::buffer<Error> allocGCErrorBuf(cl::sycl::range<1>(1));
    m_workQueue.submit([&](handler &cgh) {
      auto memPoolWrite =
          m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      auto dependencyTracker =
          m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
      auto garbageCollectorHandleAcc =
          m_garbageCollectorHandleBuff.get_access<access::mode::read>(cgh);
      auto resultValueAcc =
          m_resultValueBuff.get_access<access::mode::read>(cgh);
      auto allocGCErrorAcc =
          allocGCErrorBuf.get_access<access::mode::discard_write>(cgh);
      cgh.single_task<class init_first_block>(
          [memPoolWrite, dependencyTracker, resultValueAcc, rootNode,
           garbageCollectorHandleAcc, allocGCErrorAcc]() {
            auto gcRef =
                memPoolWrite[0].derefHandle(garbageCollectorHandleAcc[0]);
            gcRef->SetMemPoolAcc(memPoolWrite);
            const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
            RuntimeBlock_t::SharedRuntimeBlockHandle_t sharedInitialBlock;
            const auto allocError = gcRef->AllocManaged(
                sharedInitialBlock, rootNode, emptyBlock, emptyBlock,
                dependencyTracker, resultValueAcc[0], memPoolWrite,
                garbageCollectorHandleAcc[0]);
            allocGCErrorAcc[0] = allocError;
            if (allocError.GetType() == Error::Type::Success) {
              dependencyTracker[0].AddActiveBlock(sharedInitialBlock);
            }
          });
    });
    auto allocGCErrorVal = allocGCErrorBuf.get_access<access::mode::read>()[0];
    if (allocGCErrorVal.GetType() != Error::Type::Success) {
      std::stringstream errorStream;
      errorStream << "Error allocating garbage collector: "
                  << static_cast<std::underlying_type_t<Error::Type>>(
                         allocGCErrorVal.GetType())
                  << std::endl;
      throw std::runtime_error(errorStream.str());
    }
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
              depTrackerAcc[0].ResetActiveBlockCount();
            });
      });
      m_runtimeValuesRequiredCount
          .get_access<access::mode::discard_write>()[0] = 0;
      m_runtimeBlocksRequiredCount
          .get_access<access::mode::discard_write>()[0] = 0;
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
        m_workQueue.submit([&](handler &cgh) {
          auto dependencyTracker =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
          auto workingBlocksAcc =
              m_workingBlocksBuff.get_access<access::mode::read_write>(cgh);
          auto memPoolAcc =
              m_memPoolBuff.get_access<access::mode::read_write>(cgh);

          auto runtimeValuesRequiredCountAtomic =
              m_runtimeValuesRequiredCount.get_access<access::mode::atomic>(
                  cgh);
          auto runtimeBlocksRequiredCountAtomic =
              m_runtimeBlocksRequiredCount.get_access<access::mode::atomic>(
                  cgh);
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
        auto gcHandleAcc =
            m_garbageCollectorHandleBuff.get_access<access::mode::read>(cgh);
        auto runtimeValuesRequiredCountAcc =
            m_runtimeValuesRequiredCount.get_access<access::mode::read>(cgh);
        auto runtimeBlocksRequiredCountAcc =
            m_runtimeBlocksRequiredCount.get_access<access::mode::read>(cgh);
        auto requiresGarbageCollectAcc =
            m_requiresGarbageCollection.get_access<access::mode::discard_write>(
                cgh);
        cgh.single_task<class check_requires_garbage_collection>(
            [memPoolAcc, gcHandleAcc, runtimeValuesRequiredCountAcc,
             runtimeBlocksRequiredCountAcc, requiresGarbageCollectAcc] {
              const auto numRuntimeValuesFree =
                  memPoolAcc[0].GetNumFree<RuntimeBlock_t::RuntimeValue>();
              const auto numRuntimeBlocksFree =
                  memPoolAcc[0].GetNumFree<RuntimeBlock_t>();
              const auto gcNumFreeSlots =
                  memPoolAcc[0].derefHandle(gcHandleAcc[0])->GetNumFreeSlots();
              requiresGarbageCollectAcc[0] =
                  numRuntimeValuesFree < runtimeValuesRequiredCountAcc[0] ||
                  numRuntimeBlocksFree < runtimeBlocksRequiredCountAcc[0] ||
                  gcNumFreeSlots < runtimeBlocksRequiredCountAcc[0];
            });
      });

      bool requiresGarbageCollectOnHost;
      {
        auto requiresGarbageCollectAcc =
            m_requiresGarbageCollection.get_access<access::mode::read>();
        requiresGarbageCollectOnHost = requiresGarbageCollectAcc[0];
      }
      if (requiresGarbageCollectOnHost || numActiveBlocks == 0) {
        if (numActiveBlocks > 0) {
          m_workQueue.submit([&](handler &cgh) {
            auto workingBlocksAcc =
                m_workingBlocksBuff.get_access<access::mode::read_write>(cgh);
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
              m_managedAllocdCount.get_access<access::mode::discard_write>(cgh);
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
              m_managedAllocdCount.get_access<access::mode::read>();
          managedAllocdSize = managedAllocdCountHost[0];
        }
        if (managedAllocdSize > 0) {
          // Expand markings
          bool wasMarkingsExpandedInLastPass = true;
          m_markingsExpanded.get_access<access::mode::discard_write>()[0] =
              false;

          while (wasMarkingsExpandedInLastPass) {
            m_workQueue.submit([&](handler &cgh) {
              auto gcHandleAcc =
                  m_garbageCollectorHandleBuff.get_access<access::mode::read>(
                      cgh);
              auto memPoolAcc =
                  m_memPoolBuff.get_access<access::mode::read_write>(cgh);
              auto markingsExpandedAcc =
                  m_markingsExpanded.get_access<access::mode::read_write>(cgh);
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
                  m_markingsExpanded.get_access<access::mode::read_write>();
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
              m_workingBlocksBuff.get_access<access::mode::read_write>(cgh);

          auto blockErrorIdxAtomicAcc =
              m_blockErrorIdx.get_access<access::mode::atomic>(cgh);
          auto blockErrorAcc =
              m_errorsPerBlock.get_access<access::mode::read_write>(cgh);

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
                if (error.GetType() != Error::Type::Success) {
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
