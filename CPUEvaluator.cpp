#include "CPUEvaluator.h"
#include <algorithm>
#include <execution>
#include <iostream>

namespace FunGPU
{
	CPUEvaluator::CPUEvaluator(Compiler::ASTNode* rootNode): m_rootASTNode(rootNode)
	{
		m_newActiveBlocks[0] = new RuntimeBlock_t(rootNode, nullptr, nullptr, this, &m_resultValue);
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
				[&runningBlocks](const auto currentBlock) {
				currentBlock->PerformEvalPass();
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

	void CPUEvaluator::AddActiveBlock(RuntimeBlock_t* block)
	{ 
		m_newActiveBlocks.at(m_activeBlockCount++) = block;
	}
}
