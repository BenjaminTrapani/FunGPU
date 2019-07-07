#include "RuntimeBlock.h"
#include "Compiler.h"
#include "PortableMemPool.h"
#include <mutex>
#include <array>
#include <memory>

namespace FunGPU
{
	class CPUEvaluator
	{
	public:
		using RuntimeBlock_t = RuntimeBlock<CPUEvaluator>;

		CPUEvaluator(Compiler::ASTNodeHandle rootNode,
			const std::shared_ptr<PortableMemPool>& memPool);
		~CPUEvaluator();
		RuntimeBlock_t::RuntimeValue EvaluateProgram();

		void AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t& block);
	private:
		Compiler::ASTNodeHandle m_rootASTNode;
		std::shared_ptr<PortableMemPool> m_memPool;
		PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue> m_resultValue;
		std::vector<RuntimeBlock_t::SharedRuntimeBlockHandle_t> m_currentBlocks;
		std::atomic<size_t> m_activeBlockCount = 0;
		std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 4096> m_newActiveBlocks;
	};
}