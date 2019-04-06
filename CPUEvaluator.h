#include "RuntimeBlock.h"
#include "Compiler.h"
#include <mutex>

namespace FunGPU
{
	class CPUEvaluator
	{
	public:
		using RuntimeBlock_t = RuntimeBlock<CPUEvaluator>;

		CPUEvaluator(Compiler::ASTNode* rootNode);
		RuntimeBlock_t::RuntimeValue EvaluateProgram();

		void AddActiveBlock(RuntimeBlock_t* block);
	private:
		Compiler::ASTNode* m_rootASTNode;
		RuntimeBlock_t::RuntimeValue m_resultValue;
		std::vector<RuntimeBlock_t*> m_currentBlocks;
		std::mutex m_newActiveBlockMtx;
		std::vector<RuntimeBlock_t*> m_newActiveBlocks;
	};
}