#pragma once

#include "Core/PortableMemPool.hpp"
#include "Core/EvaluatorV2/Program.hpp"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/EvaluatorV2/RuntimeBlock.hpp"
#include "Core/sycl.hpp"

namespace FunGPU::EvaluatorV2 {
class Evaluator {
public:
  static constexpr Index_t REGISTERS_PER_THREAD = 64;
  static constexpr Index_t THREADS_PER_BLOCK = 32;
  using RuntimeBlock_t = RuntimeBlock<REGISTERS_PER_THREAD, THREADS_PER_BLOCK>;
  using RuntimeBlockHandle = PortableMemPool::Handle<RuntimeBlock_t>;

  Evaluator(cl::sycl::buffer<PortableMemPool>);
  RuntimeValue compute(const Program &);
  RuntimeBlockHandle construct_initial_block(const Program&);

private:
   struct IndirectCallRequest {
    FunctionValue function_val;
    PortableMemPool::Handle<RuntimeBlock_t> dest_block;
    PortableMemPool::ArrayHandle<RuntimeValue> args;
    Index_t thread_idx;
    Index_t register_idx;
  };

  class DependencyAggregator {
    public:
      RuntimeBlockHandle runtime_block(Index_t);
      void add_block(RuntimeBlockHandle);
      Index_t flip();
      Index_t num_active_blocks() const { return num_active_blocks_; }

    private:
      std::array<std::array<RuntimeBlockHandle, 4096>, 2> buffers_;
      Index_t cur_buffer_idx_ = 0;
      Index_t num_active_blocks_ = 0;
  };

  cl::sycl::buffer<PortableMemPool> mem_pool_buffer_;
  cl::sycl::buffer<IndirectCallRequest> indirect_call_requests_;
  cl::sycl::buffer<DependencyAggregator> dependency_aggregator_;
};
} // namespace FunGPU::EvaluatorV2
