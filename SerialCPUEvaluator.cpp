#include "SerialCPUEvaluator.h"

namespace FunGPU
{
	SerialCPUEvaluator::SerialCPUEvaluator(Compiler::ASTNode* rootNode): m_rootASTNode(rootNode)
	{
		m_newActiveBlocks.push_back(new RuntimeBlock_t(rootNode, nullptr, nullptr, this, &m_resultValue));
	}

	SerialCPUEvaluator::RuntimeBlock_t::RuntimeValue SerialCPUEvaluator::EvaluateProgram()
	{
		while (m_newActiveBlocks.size() > 0)
		{
			m_currentBlocks = m_newActiveBlocks;
			m_newActiveBlocks.clear();
			for (const auto currentBlock : m_currentBlocks)
			{
				currentBlock->PerformEvalPass();
			}
		}

		return m_resultValue;
	}

	void SerialCPUEvaluator::AddActiveBlock(RuntimeBlock_t* block)
	{
		m_newActiveBlocks.push_back(block);
	}
}
