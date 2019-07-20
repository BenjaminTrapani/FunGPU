#include "RuntimeBlock.h"
#include "Compiler.h"
#include "PortableMemPool.h"
#include <mutex>
#include <array>
#include <memory>
#include <SYCL\sycl.hpp>

namespace FunGPU
{
	class CPUEvaluator
	{
	public:

		class DependencyTracker;
		using RuntimeBlock_t = RuntimeBlock<DependencyTracker>;

	    class DependencyTracker
		{
		public:
			DependencyTracker() : m_activeBlockCount(0) {}
			void AddActiveBlock(const RuntimeBlock_t::SharedRuntimeBlockHandle_t& block);

		private:
			std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 4096> m_newActiveBlocks;
			std::atomic<size_t> m_activeBlockCount;
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
		cl::sycl::buffer<DependencyTracker> m_dependencyTracker;

		cl::sycl::queue m_workQueue;
	};
}