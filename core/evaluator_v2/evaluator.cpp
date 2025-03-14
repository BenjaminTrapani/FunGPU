#include "core/evaluator_v2/evaluator.hpp"
#include "core/evaluator_v2/program.hpp"
#include "core/evaluator_v2/runtime_block.hpp"
#include "core/evaluator_v2/runtime_value.hpp"
#include "core/portable_mem_pool.hpp"
#include <ostream>
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
Evaluator::Evaluator(cl::sycl::buffer<PortableMemPool> buffer)
    : mem_pool_buffer_(buffer),
      num_shared_memory_bytes_(
          work_queue_.get_device()
              .get_info<cl::sycl::info::device::local_mem_size>()) {
  std::cout << "Running on "
            << work_queue_.get_device().get_info<cl::sycl::info::device::name>()
            << ", RuntimeBlockType size: " << sizeof(RuntimeBlockType)
            << ", RuntimeValue size: " << sizeof(RuntimeValue)
            << ", Instruction size: " << sizeof(Instruction)
            << ", shared memory per block: " << num_shared_memory_bytes_
            << ", max_blocks_scheduled_per_pass=" << IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS
            << std::endl;
}

void Evaluator::check_program_does_not_overflow_shared_memory(
    const Program &program) {
  const auto num_shared_memory_bytes_required_per_block =
      static_cast<std::size_t>(max_num_instructions_in_program(
          program,
          mem_pool_buffer_.get_access<cl::sycl::access::mode::read>())) *
          sizeof(Instruction) +
      sizeof(RuntimeBlockType) + 1UZ;
  if (num_shared_memory_bytes_required_per_block > num_shared_memory_bytes_) {
    throw std::invalid_argument(
        "Program may overflow shared memory: " +
        std::to_string(num_shared_memory_bytes_required_per_block) + " > " +
        std::to_string(num_shared_memory_bytes_));
  }
}

RuntimeValue Evaluator::compute(const Program program) {
  check_program_does_not_overflow_shared_memory(program);
  IndirectCallHandlerType::Buffers indirect_call_handler_buffers(
      program.get_count());
  const auto begin_time = std::chrono::high_resolution_clock::now();
  first_block_ = construct_initial_block(program);
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    const auto first_block_tmp = first_block_;
    auto indirect_call_buffer_acc =
        indirect_call_handler_buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task([first_block_tmp, program, indirect_call_buffer_acc] {
      FunctionValue funv;
      funv.block_idx = 0;
      funv.captures = PortableMemPool::ArrayHandle<RuntimeValue>();
      IndirectCallHandlerType::on_indirect_call(
          indirect_call_buffer_acc, first_block_tmp, funv, 0, 0,
          PortableMemPool::ArrayHandle<RuntimeValue>());
    });
  });

  Index_t num_steps = 0;
  while (true) {
    const auto maybe_next_batch =
        schedule_next_batch(program, indirect_call_handler_buffers);
    if (!maybe_next_batch.has_value()) {
      break;
    }
    ++num_steps;
    run_eval_step(*maybe_next_batch, indirect_call_handler_buffers);
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  std::cout << "num_steps: " << num_steps << std::endl;
  std::cout << "total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     begin_time)
                   .count()
            << std::endl;
  return read_result(first_block_);
}

auto Evaluator::construct_initial_block(const Program program)
    -> PortableMemPool::Handle<RuntimeBlockType> {
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlockType>> initial_block_buf(
      cl::sycl::range<1>(1));
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto result_acc =
        initial_block_buf.get_access<cl::sycl::access::mode::discard_write>(
            cgh);
    auto mem_pool_acc =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task([result_acc, mem_pool_acc, program] {
      const auto &main_lambda = mem_pool_acc[0].deref_handle(program)[0];
      const auto block = mem_pool_acc[0].alloc<RuntimeBlockType>(
          main_lambda.instructions,
          RuntimeBlockType::PreAllocatedRuntimeValuesPerThread(), 1);
      result_acc[0] = block;
      auto &block_data = *mem_pool_acc[0].deref_handle(block);
      block_data.num_outstanding_dependencies = 1;
    });
  });

  return initial_block_buf.get_access<cl::sycl::access::mode::read>()[0];
}

RuntimeValue Evaluator::read_result(
    const PortableMemPool::Handle<RuntimeBlockType> first_block) {
  cl::sycl::buffer<RuntimeValue> result(cl::sycl::range<1>(1));
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto result_acc =
        result.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task([mem_pool_acc, result_acc, first_block] {
      auto &first_block_data = *mem_pool_acc[0].deref_handle(first_block);
      result_acc[0] = first_block_data.registers[0][0];
      first_block_data.deallocate_runtime_values_array_for_thread(mem_pool_acc,
                                                                  0);
      mem_pool_acc[0].dealloc(first_block);
    });
  });
  return result.get_access<cl::sycl::access::mode::read>()[0];
}

auto Evaluator::schedule_next_batch(
    const Program program,
    IndirectCallHandlerType::Buffers &indirect_call_handler_buffers)
    -> std::optional<BlockExecGroup> {
  const auto next_batch = IndirectCallHandlerType::populate_block_exec_group(
      work_queue_, mem_pool_buffer_, indirect_call_handler_buffers, program);
  if (next_batch.num_blocks > 1) {
    return next_batch;
  }
  if (next_batch.num_blocks != 1) {
    throw std::invalid_argument("Expected at least one block");
  }
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto result_acc =
        is_initial_block_ready_again_
            .get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto block_exec_acc = indirect_call_handler_buffers.block_exec_group
        .get_access<cl::sycl::access::mode::read>(cgh);
    const auto first_block_tmp = first_block_;
    cgh.single_task([block_exec_acc, next_batch, first_block_tmp, result_acc] {
      const auto &block_meta = block_exec_acc[0];
      result_acc[0] = block_meta.block == first_block_tmp;
    });
  });
  if (is_initial_block_ready_again_
          .get_access<cl::sycl::access::mode::read>()[0]) {
    return std::nullopt;
  }
  return next_batch;
}

void Evaluator::run_eval_step(
    const BlockExecGroup block_group,
    IndirectCallHandlerType::Buffers &indirect_call_handler_buffers) {
  // TODO: Even though max number of blocks per kernel launch is constrained,
  // the buffers in the memory pool and indirect call handler are shared across
  // invocations. Debug which ones are overflowing and consider using linked
  // list of blocks to more efficiently use memory. Add error reporting
  // mechanism for these failure modes.
  work_queue_.submit([block_group, &indirect_call_handler_buffers,
                      this](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto indirect_all_acc =
        indirect_call_handler_buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    auto reactivate_block_acc =
        indirect_call_handler_buffers.block_reactivation_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_exec_acc = indirect_call_handler_buffers.block_exec_group
        .get_access<cl::sycl::access::mode::read>(cgh);
    cl::sycl::accessor<RuntimeBlockType, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        local_block(cl::sycl::range<1>(1), cgh);
    cl::sycl::accessor<bool, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        any_threads_pending(cl::sycl::range<1>(1), cgh);
    RuntimeBlockType::InstructionLocalMemAccessor local_instructions(
        cl::sycl::range<2>(block_group.num_blocks,
                           block_group.max_num_instructions),
        cgh);
    cgh.parallel_for<class TestEvalLoop>(
        cl::sycl::nd_range<1>(THREADS_PER_BLOCK *
                                  block_group.num_blocks,
                              THREADS_PER_BLOCK),
        [mem_pool_write, local_block, local_instructions,
         any_threads_pending, indirect_all_acc,
         reactivate_block_acc, block_exec_acc](cl::sycl::nd_item<1> itm) {
          const auto thread_idx = itm.get_local_linear_id();
          const auto block_idx = itm.get_group_linear_id();
          const auto block_meta = block_exec_acc[block_idx];
          if (thread_idx == 0) {
            local_block[0] = *mem_pool_write[0].deref_handle(block_meta.block);
            any_threads_pending[0] = false;
          }
          const auto *instructions_global_data =
              mem_pool_write[0].deref_handle(block_meta.instructions);
          auto instructions_for_block =
              local_instructions[itm.get_group_linear_id()];
          for (auto idx = thread_idx; idx < block_meta.instructions.get_count();
               idx += THREADS_PER_BLOCK) {
            instructions_for_block[idx] = instructions_global_data[idx];
          }
          itm.barrier(cl::sycl::access::fence_space::local_space);
          const RuntimeBlockType::Status status = local_block[0].evaluate(
              itm.get_group_linear_id(), thread_idx, itm, mem_pool_write,
              local_instructions, block_meta.instructions.get_count(),
              block_meta.num_threads,
              [mem_pool_write, indirect_all_acc, reactivate_block_acc](
                  const auto block, const auto funv, const auto tid,
                  const auto reg, const auto args) {
                IndirectCallHandlerType::on_indirect_call(
                    indirect_all_acc, block, funv, tid, reg, args);
              },
              [mem_pool_write, reactivate_block_acc](const auto block) {
                IndirectCallHandlerType::on_activate_block(
                    mem_pool_write, reactivate_block_acc, block);
              });
          switch (status) {
          case RuntimeBlockType::Status::COMPLETE:
            break;
          case RuntimeBlockType::Status::STALLED:
          case RuntimeBlockType::Status::READY:
            any_threads_pending[0] = true;
            break;
          }
          itm.barrier(cl::sycl::access::fence_space::local_space);
          if (!any_threads_pending[0] && thread_idx < block_meta.num_threads) {
            mem_pool_write[0]
                .deref_handle(block_meta.block)
                ->deallocate_runtime_values_array_for_thread(mem_pool_write,
                                                             thread_idx);
          }
          cl::sycl::group_barrier(itm.get_group());
          if (thread_idx == 0) {
            if (!any_threads_pending[0]) {
              mem_pool_write[0].dealloc(block_meta.block);
            } else {
              *mem_pool_write[0].deref_handle(block_meta.block) =
                  local_block[0];
            }
          }
        });
  });
}
} // namespace FunGPU::EvaluatorV2
