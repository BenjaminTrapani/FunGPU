#include "CPUEvaluator.h"
#include <algorithm>
#include <execution>
#include <iostream>

namespace FunGPU
{
	CPUEvaluator::CPUEvaluator(const Compiler::ASTNodeHandle rootNode,
		const std::shared_ptr<PortableMemPool>& memPool): m_rootASTNode(rootNode),
		m_memPool(memPool)
	{
		m_newActiveBlocks[0] = m_memPool->Alloc<RuntimeBlock_t>(rootNode, RuntimeBlock_t::RuntimeBlockHandle_t(), 
			RuntimeBlock_t::RuntimeBlockHandle_t(), this, &m_resultValue, m_memPool);
		m_activeBlockCount = 1;
	}

	CPUEvaluator::RuntimeBlock_t::RuntimeValue CPUEvaluator::EvaluateProgram()
	{
		size_t maxConcurrentBlocks = 0;
		while (m_activeBlockCount > 0)
		{
			m_currentBlocks.clear();
			m_currentBlocks.insert(m_currentBlocks.end(), m_newActiveBlocks.begin(), 
				m_newActiveBlocks.begin() + m_activeBlockCount);
			maxConcurrentBlocks = std::max(maxConcurrentBlocks, m_currentBlocks.size());
			m_activeBlockCount = 0;

			std::atomic<size_t> runningBlocks = m_currentBlocks.size();
			std::for_each(std::execution::par_unseq, m_currentBlocks.begin(), m_currentBlocks.end(),
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
		return m_resultValue;
	}

	void CPUEvaluator::AddActiveBlock(RuntimeBlock_t::RuntimeBlockHandle_t block)
	{ 
		m_newActiveBlocks.at(m_activeBlockCount++) = block;
	}
}