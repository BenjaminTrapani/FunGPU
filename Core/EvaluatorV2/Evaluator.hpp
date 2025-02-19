#pragma once

#include "Core/EvaluatorV2/IndirectCallHandler.hpp"
#include "Core/EvaluatorV2/Program.hpp"
#include "Core/EvaluatorV2/RuntimeBlock.hpp"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/PortableMemPool.hpp"
#include "Core/sycl.hpp"
#include <CL/sycl.hpp>
#include <optional>

namespace FunGPU::EvaluatorV2 {
class Evaluator {
public:
  static constexpr Index_t REGISTERS_PER_THREAD = 16;
  static constexpr Index_t THREADS_PER_BLOCK = 32;
  using RuntimeBlockType =
      RuntimeBlock<REGISTERS_PER_THREAD, THREADS_PER_BLOCK>;
  using IndirectCallHandlerType =
      IndirectCallHandler<RuntimeBlockType, 8192 * 4, 2048>;

  Evaluator(cl::sycl::buffer<PortableMemPool>);
  RuntimeValue compute(Program);

private:
  PortableMemPool::Handle<RuntimeBlockType> construct_initial_block(Program);
  RuntimeValue read_result(PortableMemPool::Handle<RuntimeBlockType>);

  void run_eval_step(RuntimeBlockType::BlockExecGroup);
  std::optional<RuntimeBlockType::BlockExecGroup> schedule_next_batch(Program);
  void cleanup(RuntimeBlockType::BlockExecGroup);

  cl::sycl::buffer<PortableMemPool> mem_pool_buffer_;
  cl::sycl::buffer<bool> is_initial_block_ready_again_{cl::sycl::range<1>(1)};
  IndirectCallHandlerType::Buffers indirect_call_handler_buffers_;
  std::shared_ptr<IndirectCallHandlerType> indirect_call_handler_data_ =
      std::make_shared<IndirectCallHandlerType>();
  cl::sycl::buffer<IndirectCallHandlerType> indirect_call_handler_buffer_{
      indirect_call_handler_data_, cl::sycl::range<1>(1)};
  PortableMemPool::Handle<RuntimeBlockType> first_block_;
  cl::sycl::queue work_queue_;
};
} // namespace FunGPU::EvaluatorV2
