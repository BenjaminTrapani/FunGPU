#include "CPUEvaluator.h"
#include <algorithm>
#include <iostream>

using namespace cl::sycl;

namespace FunGPU
{
	class init_first_block;
    class run_eval_pass;

	void CPUEvaluator::DependencyTracker::AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t& block)
	{
		m_newActiveBlocks[m_activeBlockCount++] = block;
	}

	CPUEvaluator::CPUEvaluator(cl::sycl::buffer<PortableMemPool> memPool):
		m_dependencyTracker(std::make_shared<DependencyTracker>()),
		m_memPoolBuff(memPool),
        m_dependencyTrackerBuff(m_dependencyTracker, range<1>(1))
	{
        std::cout << std::endl;
        std::cout << "Running on "
                  << m_workQueue.get_device().get_info<info::device::name>()
                  << std::endl;

        {
            auto memPoolHostAcc = m_memPoolBuff.get_access<access::mode::read_write>();
            m_resultValue = memPoolHostAcc[0].template Alloc<RuntimeBlock_t::RuntimeValue>();
        }
	}

	CPUEvaluator::~CPUEvaluator()
	{
        auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>();
        memPoolAcc[0].Dealloc(m_resultValue);
	}

	void CPUEvaluator::CreateFirstBlock(const Compiler::ASTNodeHandle rootNode) {
        try {
            auto resultValueRefCpy = m_resultValue;
            m_workQueue.submit([&](handler &cgh) {
                auto memPoolWrite = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
                auto dependencyTracker = m_dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
                cgh.single_task<class init_first_block>(
                        [memPoolWrite, dependencyTracker, resultValueRefCpy, rootNode]() {
                            const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
                            const auto sharedInitialBlock = memPoolWrite[0].Alloc<RuntimeBlock_t>(rootNode,
                                                                                                  emptyBlock,
                                                                                                  emptyBlock,
                                                                                                  dependencyTracker,
                                                                                                  resultValueRefCpy,
                                                                                                  memPoolWrite);
                            dependencyTracker[0].AddActiveBlock(sharedInitialBlock);
                        });
            });
            m_workQueue.wait();
        } catch (cl::sycl::exception e) {
            std::cerr << "Sycl exception: " << e.what() << std::endl;
        }
	}

	CPUEvaluator::RuntimeBlock_t::RuntimeValue CPUEvaluator::EvaluateProgram(const Compiler::ASTNodeHandle& rootNode,
                                                                             unsigned int& maxConcurrentBlocksDuringExec)
	{
        CreateFirstBlock(rootNode);

        maxConcurrentBlocksDuringExec = 0;
        try {
            buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> workingBlocksBuff(range<1>(m_dependencyTracker->m_newActiveBlocks.size()));
            while (true) {
                unsigned int numActiveBlocks = 0;
                {
                    auto hostDepTracker = m_dependencyTrackerBuff.get_access<access::mode::read_write>();
                    numActiveBlocks = hostDepTracker[0].GetActiveBlockCount();
                    hostDepTracker[0].ResetActiveBlockCount();
                    auto workingBlocksAcc = workingBlocksBuff.get_access<access::mode::read_write>();
                    //TODO make this copy parallel, do it on device.
                    for (size_t i = 0; i < numActiveBlocks; ++i)
                    {
                        workingBlocksAcc[i] = hostDepTracker[0].GetBlockAtIndex(i);
                    }
                }

                maxConcurrentBlocksDuringExec = std::max(numActiveBlocks, maxConcurrentBlocksDuringExec);

                if (numActiveBlocks > 0) {
                    m_workQueue.submit([&](handler &cgh) {
                        auto memPoolWrite = m_memPoolBuff.get_access<access::mode::read_write>(cgh);
                        auto dependencyTracker = m_dependencyTrackerBuff.get_access<access::mode::read_write>(
                                cgh);
                        auto workingBlocksAcc = workingBlocksBuff.get_access<access::mode::read>(cgh);
                        cgh.parallel_for<class run_eval_pass>(cl::sycl::range<1>(numActiveBlocks),
                                                              [dependencyTracker, memPoolWrite, workingBlocksAcc](
                                                                      item<1> itm) {
                                                                  auto currentBlock = workingBlocksAcc[itm.get_linear_id()];
                                                                  auto derefdCurrentBlock = memPoolWrite[0].derefHandle(
                                                                          currentBlock);
                                                                  derefdCurrentBlock->SetMemPool(memPoolWrite);
                                                                  derefdCurrentBlock->SetDependencyTracker(dependencyTracker);
                                                                  derefdCurrentBlock->PerformEvalPass();
                                                              });
                    });
                } else {
                    break;
                }
                m_workQueue.wait();
            }
        } catch (cl::sycl::exception e) {
            std::cerr << "Sycl exception: " << e.what() << std::endl;
        }

		auto memPoolAcc = m_memPoolBuff.get_access<access::mode::read_write>();
		return *memPoolAcc[0].derefHandle(m_resultValue);
	}
}
