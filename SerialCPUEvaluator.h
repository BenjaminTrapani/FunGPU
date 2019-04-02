#include "RuntimeBlock.h"
#include "Compiler.h"

namespace FunGPU
{
	class SerialCPUEvaluator
	{
	public:
		using RuntimeBlock_t = RuntimeBlock<SerialCPUEvaluator>;

		SerialCPUEvaluator(Compiler::ASTNode* rootNode);
		RuntimeBlock_t::RuntimeValue EvaluateProgram();

		void AddActiveBlock(RuntimeBlock_t* block);
	private:
		Compiler::ASTNode* m_rootASTNode;
		RuntimeBlock_t::RuntimeValue m_resultValue;
		std::vector<RuntimeBlock_t*> m_currentBlocks;
		std::vector<RuntimeBlock_t*> m_newActiveBlocks;
	};
}