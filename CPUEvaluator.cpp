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

	CPUEvaluator::CPUEvaluator(const Compiler::ASTNodeHandle rootNode,
		const std::shared_ptr<PortableMemPool>& memPool): m_rootASTNode(rootNode),
		m_memPool(memPool),
		m_dependencyTracker(std::make_shared<DependencyTracker>())
	{
	    std::cout << std::endl;
		std::cout << "Running on "
			<< m_workQueue.get_device().get_info<info::device::name>()
			<< std::endl;

		try {
            m_resultValue = m_memPool->Alloc<RuntimeBlock_t::RuntimeValue>();
            {
                buffer<PortableMemPool> memPoolBuff(m_memPool, range<1>(1));
                buffer<DependencyTracker> dependencyTrackerBuff(m_dependencyTracker, range<1>(1));
                buffer<RuntimeBlock_t::SharedRuntimeBlockHandle_t> workingBlocksBuff(range<1>(m_dependencyTracker->m_newActiveBlocks.size()));
                auto resultValueRefCpy = m_resultValue;
                m_workQueue.submit([&](handler &cgh) {
                    auto memPoolWrite = memPoolBuff.get_access<access::mode::read_write>(cgh);
                    auto dependenyTracker = dependencyTrackerBuff.get_access<access::mode::read_write>(cgh);
                    cgh.single_task<class init_first_block>(
                            [memPoolWrite, dependenyTracker, resultValueRefCpy, rootNode]() {
                                const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
                                const auto sharedInitialBlock = memPoolWrite[0].Alloc<RuntimeBlock_t>(rootNode,
                                                                                                      emptyBlock,
                                                                                                      emptyBlock,
                                                                                                      dependenyTracker,
                                                                                                      resultValueRefCpy,
                                                                                                      memPoolWrite);
                                dependenyTracker[0].AddActiveBlock(sharedInitialBlock);
                            });
                });

                while (true) {
                    unsigned int numActiveBlocks = 0;
                    {
                        auto hostDepTracker = dependencyTrackerBuff.get_access<access::mode::read_write>();
                        numActiveBlocks = hostDepTracker[0].GetActiveBlockCount();
                        hostDepTracker[0].ResetActiveBlockCount();
                        auto workingBlocksAcc = workingBlocksBuff.get_access<access::mode::read_write>();
                        //TODO make this copy parallel, do it on device.
                        for (size_t i = 0; i < numActiveBlocks; ++i)
                        {
                            workingBlocksAcc[i] = hostDepTracker[0].GetBlockAtIndex(i);
                        }
                    }

                    if (numActiveBlocks > 0) {
                        m_workQueue.submit([&](handler &cgh) {
                            auto memPoolWrite = memPoolBuff.get_access<access::mode::read_write>(cgh);
                            auto dependenyTracker = dependencyTrackerBuff.get_access<access::mode::read_write>(
                                    cgh);
                            auto workingBlocksAcc = workingBlocksBuff.get_access<access::mode::read>(cgh);
                            cgh.parallel_for<class run_eval_pass>(cl::sycl::range<1>(numActiveBlocks),
                                                                  [dependenyTracker, memPoolWrite, workingBlocksAcc](
                                                                          item<1> itm) {
                                                                      auto currentBlock = workingBlocksAcc[itm.get_linear_id()];
                                                                      auto derefdCurrentBlock = memPoolWrite[0].derefHandle(
                                                                              currentBlock);
                                                                      derefdCurrentBlock->SetMemPool(memPoolWrite);
                                                                      derefdCurrentBlock->PerformEvalPass();
                                                                  });
                            std::cout << "Completed eval pass" << std::endl;
                        });
                    } else {
                        break;
                    }
                    m_workQueue.wait();
                }
            }
        } catch (cl::sycl::exception e) {
		    std::cerr << "Sycl exception: " << e.what() << std::endl;
		}
		m_workQueue.wait();
	}

	CPUEvaluator::~CPUEvaluator()
	{
		m_memPool->Dealloc(m_resultValue);
	}

	CPUEvaluator::RuntimeBlock_t::RuntimeValue CPUEvaluator::EvaluateProgram()
	{
		/*size_t maxConcurrentBlocks = 0;
		while (m_activeBlockCount > 0)
		{
			m_currentBlocks.clear();
			m_currentBlocks.insert(m_currentBlocks.end(), m_newActiveBlocks.begin(), 
				m_newActiveBlocks.begin() + m_activeBlockCount);
			maxConcurrentBlocks = std::max(maxConcurrentBlocks, m_currentBlocks.size());
			m_activeBlockCount = 0;

			std::atomic<size_t> runningBlocks; 
			runningBlocks.store(m_currentBlocks.size());

			std::for_each(m_currentBlocks.begin(), m_currentBlocks.end(),
				[&runningBlocks, this](const auto currentBlock) {
				auto derefdCurrentBlock = m_memPool->derefHandle(currentBlock);
				derefdCurrentBlock->PerformEvalPass();
				--runningBlocks;
			});
			while (runningBlocks > 0)
			{
			}
		}
		std::cout << std::endl;
		std::cout << "Max concurrent blocks during exec: " << maxConcurrentBlocks << std::endl;
		*/
		return *m_memPool->derefHandle(m_resultValue);
	}
}
