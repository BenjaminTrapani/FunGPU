#include "RuntimeBlock.h"
#include "Compiler.h"
#include "PortableMemPool.h"
#include <array>
#include <memory>
#include <SYCL/sycl.hpp>

namespace FunGPU
{
	class CPUEvaluator
	{
	public:

		class DependencyTracker;
		using RuntimeBlock_t = RuntimeBlock<DependencyTracker>;

	    class DependencyTracker
        {
	        friend class CPUEvaluator;
		public:
			DependencyTracker() : m_activeBlockCount(0) {}
			void AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t& block);
            unsigned int GetActiveBlockCount() {
                return m_activeBlockCount.load();
            }
            void ResetActiveBlockCount() {
                m_activeBlockCount.store(0);
            }
            RuntimeBlock_t::SharedRuntimeBlockHandle_t GetBlockAtIndex(const unsigned int index) {
                return m_newActiveBlocks[index];
            }
		private:
			std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 4096> m_newActiveBlocks;
			std::atomic<unsigned int> m_activeBlockCount;
		};

		CPUEvaluator(Compiler::ASTNodeHandle rootNode,
			const std::shared_ptr<PortableMemPool>& memPool);
		~CPUEvaluator();
		RuntimeBlock_t::RuntimeValue EvaluateProgram();

	private:
		Compiler::ASTNodeHandle m_rootASTNode;
		std::shared_ptr<PortableMemPool> m_memPool;
		PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue> m_resultValue;
		std::vector<RuntimeBlock_t::SharedRuntimeBlockHandle_t> m_currentBlocks;
		std::shared_ptr<DependencyTracker> m_dependencyTracker;

		cl::sycl::queue m_workQueue;
	};
}