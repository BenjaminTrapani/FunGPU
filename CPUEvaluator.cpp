#include "CPUEvaluator.h"
#include <algorithm>
#include <iostream>

using namespace cl::sycl;

namespace FunGPU
{
	class init_first_block;

	void CPUEvaluator::DependencyTracker::AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t& block)
	{
		m_newActiveBlocks[m_activeBlockCount++] = block;
	}

	CPUEvaluator::CPUEvaluator(const Compiler::ASTNodeHandle rootNode,
		const std::shared_ptr<PortableMemPool>& memPool): m_rootASTNode(rootNode),
		m_memPool(memPool),
		m_dependencyTracker(cl::sycl::range<1>(1))
	{
		std::cout << "Running on "
			<< m_workQueue.get_device().get_info<info::device::name>()
			<< std::endl;

		// Buffer is not a standard layout class, so can't have it as member on PortableMemPool.
		// TODO create a PortableMemPoolStorage block that contains the buffers and data structures. Provide 
		// templated accessors for PortableMemPoolStorage that contain most of the logic in mem pool now but 
		// that create either device or host accessors that last for the duration of their instance based on the input
		// mem pool storage block. Create the PortableMemPool instance here. Since it only contains accessors
		// and no large blocks of data, it is trivially copyable so no need to muck with passing around
		// references to the mem pool. Can simply pass around copies of the one backed by the storage.
		m_resultValue = m_memPool->Alloc<cl::sycl::access::target::host_buffer, RuntimeBlock_t::RuntimeValue>();

		buffer<PortableMemPool> memPoolBuff(m_memPool.get(), range<1>(1));

		auto resultValueRefCpy = m_resultValue;
		m_workQueue.submit([&](handler& cgh) {
			{
				auto memPoolWriteHost = memPoolBuff.get_access<access::mode::read_write, access::target::host_buffer>(cgh);
				memPoolWriteHost[0].ConfigureForDeviceAccess(cgh);
			}

			auto memPoolWrite = memPoolBuff.get_access<access::mode::read_write>(cgh);
			auto dependenyTracker = m_dependencyTracker.get_access<access::mode::read_write>(cgh);

			/*const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
			PortableMemPool* memPoolRef = &memPoolWrite[0];
			const auto sharedInitialBlock = memPoolWrite[0].AllocShared<RuntimeBlock_t>(rootNode, emptyBlock,
				emptyBlock, dependencyTracker, resultValueAcc[0], memPoolRef);
			newActiveBlocksWrite[0] = sharedInitialBlock;*/

			cgh.single_task<class init_first_block> ([&cgh, memPoolWrite, dependenyTracker, resultValueRefCpy, rootNode]() {
				const RuntimeBlock_t::SharedRuntimeBlockHandle_t emptyBlock;
				const auto sharedInitialBlock = memPoolWrite[0].AllocShared<RuntimeBlock_t>(rootNode, emptyBlock,
					emptyBlock, dependenyTracker, resultValueRefCpy, memPoolWrite, &cgh);
				dependenyTracker[0].AddActiveBlock(sharedInitialBlock);
			});

			//memPoolWrite[0].SetDeviceHandler(cgh);
		});
		
		m_workQueue.wait();

		//m_memPool->ClearDeviceHandler();
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
