#pragma once

#include "core/evaluator_v2/indirect_call_handler.hpp"
#include "core/evaluator_v2/program.hpp"
#include "core/evaluator_v2/runtime_block.hpp"
#include "core/evaluator_v2/runtime_value.hpp"
#include "core/portable_mem_pool.hpp"
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

  void run_eval_step(RuntimeBlockType::BlockExecGroup,
                     IndirectCallHandlerType::Buffers &);
  std::optional<RuntimeBlockType::BlockExecGroup>
  schedule_next_batch(Program, IndirectCallHandlerType::Buffers &);
  void cleanup(RuntimeBlockType::BlockExecGroup);

  cl::sycl::buffer<PortableMemPool> mem_pool_buffer_;
  cl::sycl::buffer<bool> is_initial_block_ready_again_{cl::sycl::range<1>(1)};
  std::shared_ptr<IndirectCallHandlerType> indirect_call_handler_data_ =
      std::make_shared<IndirectCallHandlerType>();
  cl::sycl::buffer<IndirectCallHandlerType> indirect_call_handler_buffer_{
      indirect_call_handler_data_, cl::sycl::range<1>(1)};
  PortableMemPool::Handle<RuntimeBlockType> first_block_;
  cl::sycl::queue work_queue_;
};
} // namespace FunGPU::EvaluatorV2
