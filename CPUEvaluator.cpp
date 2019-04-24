#include "CPUEvaluator.h"
#include <algorithm>
#include <execution>

namespace FunGPU
{
	CPUEvaluator::CPUEvaluator(Compiler::ASTNode* rootNode): m_rootASTNode(rootNode)
	{
		std::lock_guard<std::mutex> guard(m_newActiveBlockMtx);
		m_newActiveBlocks.push_back(new RuntimeBlock_t(rootNode, nullptr, nullptr, this, &m_resultValue));
	}

	CPUEvaluator::RuntimeBlock_t::RuntimeValue CPUEvaluator::EvaluateProgram()
	{
		while (m_newActiveBlocks.size() > 0 || m_resultValue.m_type == RuntimeBlock_t::RuntimeValue::Type::Lazy)
		{
			{
				std::lock_guard<std::mutex> guard(m_newActiveBlockMtx);
				if (m_newActiveBlocks.size() == 0)
				{
					m_currentBlocks = { m_resultValue.m_data.lazyVal };
				}
				else
				{
					m_currentBlocks = m_newActiveBlocks;
					m_newActiveBlocks.clear();
				}
			}

			size_t runningBlocks = m_currentBlocks.size();
			std::mutex runningBlockMtx;
			std::condition_variable cv;
			std::for_each(std::execution::par_unseq, m_currentBlocks.begin(), m_currentBlocks.end(),
				[&runningBlocks, &runningBlockMtx, &cv](const auto currentBlock) {
				currentBlock->PerformEvalPass();
				{
					std::lock_guard<std::mutex> guard(runningBlockMtx);
					--runningBlocks;
				}
				cv.notify_one();
			});
			std::unique_lock<std::mutex> lk(runningBlockMtx);
			cv.wait(lk, [&runningBlocks] {return runningBlocks == 0; });
		}

		return m_resultValue;
	}

	void CPUEvaluator::AddActiveBlock(RuntimeBlock_t* block)
	{
		std::lock_guard<std::mutex> guard(m_newActiveBlockMtx);
		m_newActiveBlocks.push_back(block);
	}
}
