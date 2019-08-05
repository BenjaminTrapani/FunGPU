#include "CPUEvaluator.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cl::sycl;

namespace FunGPU {
class init_first_block;
class run_eval_pass;
class prep_blocks;
class mark_dependent_blocks;
class prep_next_marking_pass;
class sweep_unmarked;
class compact_alive_blocks;
class write_compacted_blocks;

void CPUEvaluator::DependencyTracker::AddActiveBlock(
    const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block,
    const PortableMemPool::DeviceAccessor_t &memPoolAcc) {

  memPoolAcc[0].AddRef(block);
  m_newActiveBlocks[m_activeBlockCount++] = block;
}

void CPUEvaluator::DependencyTracker::MarkRequiresRemoveRef(
    const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block) {
  m_requireRemoveRefBlocks[m_removeRefBlocksCount++] = block;
}

CPUEvaluator::CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool)
    : m_dependencyTracker(std::make_shared<DependencyTracker>()),
      m_memPoolBuff(memPool),
      m_dependencyTrackerBuff(m_dependencyTracker, range<1>(1)),
m_workQueue(host_selector{}) {
  std::cout << std::endl;
  std::cout << "Running on "
            << m_workQueue.get_device().get_info<info::device::name>()
            << std::endl;

  {
    auto memPoolHostAcc = m_memPoolBuff.get_access<access::mode::read_write>();
    m_resultValue =
        memPoolHostAcc[0].template Alloc<RuntimeBlock_t::RuntimeValue>();
  }
}

CPUEvaluator::~CPUEvaluator() {
  auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>();
  memPoolAcc[0].Dealloc(m_resultValue);
}

void CPUEvaluator::CreateFirstBlock(const Compiler::ASTNodeHandle rootNode) {
  try {
    auto resultValueRefCpy = m_resultValue;
    m_workQueue.submit([&](handler &cgh) {
      auto memPoolWrite =
          m_memPoolBuff.get_access<access::mode::read_write>(cgh);
      auto dependencyTracker =
          m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class init_first_block>([memPoolWrite, dependencyTracker,
                                               resultValueRefCpy, rootNode]() {
        const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
        const auto sharedInitialBlock =
            memPoolWrite[0].AllocShared<RuntimeBlock_t>(
                rootNode, emptyBlock, emptyBlock, dependencyTracker,
                resultValueRefCpy, memPoolWrite);
        dependencyTracker[0].AddActiveBlock(sharedInitialBlock, memPoolWrite);
        memPoolWrite[0].RemoveRef(sharedInitialBlock);
      });
    });
    m_workQueue.wait();
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
CPUEvaluator::EvaluateProgram(const Compiler::ASTNodeHandle &rootNode,
                              unsigned int &maxConcurrentBlocksDuringExec) {
  CreateFirstBlock(rootNode);

  maxConcurrentBlocksDuringExec = 0;
  try {
    buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> workingBlocksBuff(
        range<1>(m_dependencyTracker->m_newActiveBlocks.size()));

    buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> aliveBlocksBuff(
            range<1>(8192));
    buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> compactedAliveBlocksBuff(
            range<1>(aliveBlocksBuff.get_size()));

    unsigned int aliveBlockCountData = 0;
    buffer<unsigned int> aliveBlockCount(&aliveBlockCountData, range<1>(1));

    buffer<RuntimeBlock_t::Error> errorsPerBlock(
        range<1>(m_dependencyTracker->m_newActiveBlocks.size()));
    unsigned int blockErrorIdxData = 0;
    buffer<unsigned int> blockErrorIdx(&blockErrorIdxData, range<1>(1));
    while (true) {
      unsigned int numActiveBlocks = 0;
      {
        auto hostDepTracker =
            m_dependencyTrackerBuff.get_access<access::mode::read_write>();
        numActiveBlocks = hostDepTracker[0].GetActiveBlockCount();
        hostDepTracker[0].ResetActiveBlockCount();

        auto hostBlockHasErrorAcc =
            blockErrorIdx.get_access<access::mode::read_write>();
        if (hostBlockHasErrorAcc[0] > 0) {
          auto errorsPerBlockAcc =
              errorsPerBlock.get_access<access::mode::read_write>();
          std::stringstream ss;
          ss << "Block errors in previous pass: " << std::endl;
          for (unsigned int i = 0; i < hostBlockHasErrorAcc[0]; ++i) {
            ss << errorsPerBlockAcc[i].GetDescription() << std::endl;
          }
          throw std::runtime_error(ss.str());
        }
        hostBlockHasErrorAcc[0] = 0;
      }

      maxConcurrentBlocksDuringExec =
          std::max(numActiveBlocks, maxConcurrentBlocksDuringExec);

      if (numActiveBlocks > 0) {

        m_workQueue.submit([&](handler &cgh) {
            auto aliveBlockCountAcc = aliveBlockCount.get_access<cl::sycl::access::mode::read_write>();
            const unsigned int prevAliveBlockCount = aliveBlockCountAcc[0];
            aliveBlockCountAcc[0] += numActiveBlocks;

          auto dependencyTracker =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
          auto workingBlocksAcc =
              workingBlocksBuff.get_access<access::mode::read_write>(cgh);
          auto aliveBlocksAcc =
                  aliveBlocksBuff.get_access<access::mode::read_write>(cgh);
          cgh.parallel_for<class prep_blocks>(
              cl::sycl::range<1>(numActiveBlocks),
              [dependencyTracker, workingBlocksAcc, prevAliveBlockCount, aliveBlocksAcc](item<1> itm) {
                const auto idx = itm.get_linear_id();
                workingBlocksAcc[idx] =
                    dependencyTracker[0].GetBlockAtIndex(idx);
                aliveBlocksAcc[idx + prevAliveBlockCount] = dependencyTracker[0].GetBlockAtIndex(idx);
              });
        });

        // Collect garbage, mark phase
        unsigned int markedBlockCount = numActiveBlocks;
        while (markedBlockCount > 0)
        {
            m_workQueue.submit([&](handler &cgh) {
                auto workingBlocksAcc =
                        workingBlocksBuff.get_access<access::mode::read_write>(cgh);
                auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
                cgh.parallel_for<class mark_dependent_blocks>(
                        cl::sycl::range<1>(markedBlockCount),
                        [workingBlocksAcc, memPoolAcc](item<1> itm) {
                            const auto idx = itm.get_linear_id();
                            auto derefdWorkingBlock = memPoolAcc[0].derefHandle(workingBlocksAcc[idx]);
                            derefdWorkingBlock->MarkDependencies();
                        });
            });
            m_workQueue.wait();

            {
                auto depTrackerHost =
                        m_dependencyTrackerBuff.get_access<access::mode::read_write>();
                markedBlockCount =
                        depTrackerHost[0].GetRequiresRemoveRefCount();
                depTrackerHost[0].ResetRequiresRemoveRefCount();
            };

            if (markedBlockCount > 0) {
                m_workQueue.submit([&](handler &cgh) {
                    auto dependencyTracker =
                            m_dependencyTrackerBuff.get_access<access::mode::read_write>(
                                    cgh);
                    auto workingBlocksAcc =
                            workingBlocksBuff.get_access<access::mode::read_write>(cgh);

                    cgh.parallel_for<class prep_next_marking_pass>(
                            cl::sycl::range<1>(markedBlockCount),
                            [dependencyTracker, workingBlocksAcc](item<1> itm) {
                                const auto idx = itm.get_linear_id();
                                workingBlocksAcc[idx] =
                                        dependencyTracker[0].GetRemoveRefBlockAtIndex(idx);
                            });
                });
            }
            m_workQueue.wait();
        };

        // Collect garbage, sweep stage
        const unsigned int curAliveBlockCount = aliveBlockCount.get_access<access::mode::read>()[0];
        m_workQueue.submit([&](handler &cgh){
            auto aliveBlocksAcc =
                    aliveBlocksBuff.get_access<access::mode::read_write>(cgh);
            auto memPoolWrite =
                    m_memPoolBuff.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class sweep_unmarked>(
                    cl::sycl::range<1>(curAliveBlockCount),
                    [aliveBlocksAcc, memPoolWrite](item<1> itm) {
                        const auto idx = itm.get_linear_id();
                        auto blockData = memPoolWrite[0].derefHandle(aliveBlocksAcc[idx]);
                        if (!blockData->GetIsMarked())
                        {
                            memPoolWrite[0].Dealloc(aliveBlocksAcc[idx]);
                            aliveBlocksAcc[idx] = RuntimeBlock_t::SharedRuntimeBlockHandle_t();
                        }
                        else {
                            blockData->ClearIsMarked();
                        }
                    });
        });

        // Compact alive blocks
        m_workQueue.submit([&](handler &cgh) {
            {
                auto aliveBlockCountAcc = aliveBlockCount.get_access<access::mode::write>();
                aliveBlockCountAcc[0] = 0;
            }
            auto aliveBlockCountAccAtomic = aliveBlockCount.get_access<access::mode::atomic>(cgh);
            auto aliveBlocksAcc =
                    aliveBlocksBuff.get_access<access::mode::read_write>(cgh);
            auto aliveBlockCompactedAcc = compactedAliveBlocksBuff.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class compact_alive_blocks>(cl::sycl::range<1>(curAliveBlockCount),
                [aliveBlockCountAccAtomic, aliveBlocksAcc, aliveBlockCompactedAcc](item<1> itm) {
                    const auto idx = itm.get_linear_id();
                    if (aliveBlocksAcc[idx] != RuntimeBlock_t::SharedRuntimeBlockHandle_t())
                    {
                        cl::sycl::atomic<unsigned int> aliveCountAtomic(aliveBlockCountAccAtomic[0]);
                        aliveBlockCompactedAcc[aliveCountAtomic.fetch_add(1)] = aliveBlocksAcc[idx];
                    }
                });
        });

        // assign compacted blocks to alive blocks
          m_workQueue.submit([&](handler &cgh) {
              auto updatedAliveBlockCount = aliveBlockCount.get_access<access::mode::read>()[0];
              auto aliveBlocksAcc =
                      aliveBlocksBuff.get_access<access::mode::write>(cgh);
              auto aliveBlockCompactedAcc = compactedAliveBlocksBuff.get_access<access::mode::read>(cgh);
              cgh.parallel_for<class write_compacted_blocks>(cl::sycl::range<1>(updatedAliveBlockCount),
                                                           [aliveBlocksAcc, aliveBlockCompactedAcc](
                                                                   item<1> itm) {
                                                               const auto idx = itm.get_linear_id();
                                                               aliveBlocksAcc[idx] = aliveBlockCompactedAcc[idx];
                                                           });
          });

        m_workQueue.submit([&](handler &cgh) {
          auto memPoolWrite =
              m_memPoolBuff.get_access<access::mode::read_write>(cgh);
          auto dependencyTracker =
              m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
          auto workingBlocksAcc =
              workingBlocksBuff.get_access<access::mode::read_write>(cgh);

          auto blockErrorIdxAtomicAcc =
              blockErrorIdx.get_access<access::mode::atomic>(cgh);
          auto blockErrorAcc =
              errorsPerBlock.get_access<access::mode::read_write>(cgh);

          cgh.parallel_for<class run_eval_pass>(
              cl::sycl::range<1>(numActiveBlocks),
              [dependencyTracker, memPoolWrite, workingBlocksAcc,
               blockErrorIdxAtomicAcc, blockErrorAcc](item<1> itm) {
                auto currentBlock = workingBlocksAcc[itm.get_linear_id()];
                auto derefdCurrentBlock =
                    memPoolWrite[0].derefHandle(currentBlock);
                derefdCurrentBlock->SetMemPool(memPoolWrite);
                derefdCurrentBlock->SetDependencyTracker(dependencyTracker);
                const auto error = derefdCurrentBlock->PerformEvalPass();
                if (error.GetType() != RuntimeBlock_t::Error::Type::Success) {
                  cl::sycl::atomic<unsigned int> blockErrorIdxAtomic(
                      blockErrorIdxAtomicAcc[0]);
                  blockErrorAcc[blockErrorIdxAtomic.fetch_add(1)] = error;
                }
              });
        });

        m_workQueue.wait();
        std::cout << "Finished collection pass" << std::endl;
      } else {
        break;
      }
    }
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }

  auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>();
  return *memPoolAcc[0].derefHandle(m_resultValue);
}
}
