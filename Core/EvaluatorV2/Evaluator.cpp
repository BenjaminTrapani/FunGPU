#include "Core/EvaluatorV2/Evaluator.hpp"
#include "Core/EvaluatorV2/RuntimeBlock.hpp"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/PortableMemPool.hpp"
#include <ostream>
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
Evaluator::Evaluator(cl::sycl::buffer<PortableMemPool> buffer)
    : mem_pool_buffer_(buffer), indirect_call_handler_buffers_(4) {
  std::cout << "Running on "
            << work_queue_.get_device().get_info<cl::sycl::info::device::name>()
            << ", block size: " << sizeof(RuntimeBlockType)
            << ", RuntimeBlockType::target_data size: "
            << sizeof(RuntimeBlockType::target_data)
            << ", runtime value size: " << sizeof(RuntimeValue) << std::endl;
}

RuntimeValue Evaluator::compute(const Program program) {
  indirect_call_handler_buffers_ =
      IndirectCallHandlerType::Buffers(program.get_count());
  const auto begin_time = std::chrono::high_resolution_clock::now();
  first_block_ = construct_initial_block(program);
  indirect_call_handler_buffers_.update_for_num_lambdas(program.get_count());
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto indirect_call_acc =
        indirect_call_handler_buffer_
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    const auto first_block_tmp = first_block_;
    auto indirect_call_buffer_acc =
        indirect_call_handler_buffers_.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task([indirect_call_acc, first_block_tmp, program,
                     indirect_call_buffer_acc] {
      FunctionValue funv;
      funv.block_idx = 0;
      funv.captures = PortableMemPool::ArrayHandle<RuntimeValue>();
      indirect_call_acc[0].on_indirect_call(
          indirect_call_buffer_acc, first_block_tmp, funv, 0, 0,
          PortableMemPool::ArrayHandle<RuntimeValue>());
    });
  });

  Index_t num_steps = 0;
  while (true) {
    const auto maybe_next_batch = schedule_next_batch(program);
    if (!maybe_next_batch.has_value()) {
      break;
    }
    ++num_steps;
    run_eval_step(*maybe_next_batch);
    cleanup(*maybe_next_batch);
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

void Evaluator::cleanup(const RuntimeBlockType::BlockExecGroup exec_group) {
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task([mem_pool_acc, exec_group] {
      mem_pool_acc[0].dealloc_array(exec_group.block_descs);
    });
  });
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

auto Evaluator::schedule_next_batch(const Program program)
    -> std::optional<RuntimeBlockType::BlockExecGroup> {
  const auto next_batch = IndirectCallHandlerType::create_block_exec_group(
      work_queue_, mem_pool_buffer_, indirect_call_handler_buffer_,
      indirect_call_handler_buffers_, program);
  if (next_batch.block_descs.get_count() > 1) {
    return next_batch;
  }
  if (next_batch.block_descs.get_count() != 1) {
    throw std::invalid_argument("Expected at least one block");
  }
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto result_acc =
        is_initial_block_ready_again_
            .get_access<cl::sycl::access::mode::discard_write>(cgh);
    const auto first_block_tmp = first_block_;
    cgh.single_task([mem_pool_acc, next_batch, first_block_tmp, result_acc] {
      const auto &block_meta =
          mem_pool_acc[0].deref_handle((next_batch.block_descs))[0];
      result_acc[0] = block_meta.block == first_block_tmp;
    });
  });
  if (is_initial_block_ready_again_
          .get_access<cl::sycl::access::mode::read>()[0]) {
    cleanup(next_batch);
    return std::nullopt;
  }
  return next_batch;
}

void Evaluator::run_eval_step(
    const RuntimeBlockType::BlockExecGroup block_group) {
  constexpr Index_t MAX_NUM_BLOCKS_PER_LAUNCH = 64;
  // TODO: Even though max number of blocks per kernel launch is constrained,
  // the buffers in the memory pool and indirect call handler are shared across
  // invocations. Debug which ones are overflowing and consider using linked
  // list of blocks to more efficiently use memory. Add error reporting
  // mechanism for these failure modes.
  for (Index_t num_launched = 0;
       num_launched < block_group.block_descs.get_count();
       num_launched += MAX_NUM_BLOCKS_PER_LAUNCH) {
    const auto num_blocks_for_this_launch =
        std::min(block_group.block_descs.get_count() - num_launched,
                 MAX_NUM_BLOCKS_PER_LAUNCH);
    const auto tmp_num_launched = num_launched;
    work_queue_.submit([num_blocks_for_this_launch, block_group,
                        tmp_num_launched, this](cl::sycl::handler &cgh) {
      auto mem_pool_write =
          mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto indirect_call_handler_acc =
          indirect_call_handler_buffer_
              .get_access<cl::sycl::access::mode::read_write>(cgh);
      auto indirect_all_acc =
          indirect_call_handler_buffers_.indirect_call_requests_by_block
              .get_access<cl::sycl::access::mode::read_write>(cgh);
      cl::sycl::accessor<RuntimeBlockType, 1,
                         cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          local_block(cl::sycl::range<1>(1), cgh);
      cl::sycl::accessor<bool, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          any_threads_pending(cl::sycl::range<1>(1), cgh);
      RuntimeBlockType::InstructionLocalMemAccessor local_instructions(
          cl::sycl::range<2>(num_blocks_for_this_launch,
                             block_group.max_num_instructions),
          cgh);
      cgh.parallel_for<class TestEvalLoop>(
          cl::sycl::nd_range<1>(THREADS_PER_BLOCK * num_blocks_for_this_launch,
                                THREADS_PER_BLOCK),
          [mem_pool_write, block_group, local_block, local_instructions,
           indirect_call_handler_acc, any_threads_pending, tmp_num_launched,
           indirect_all_acc](cl::sycl::nd_item<1> itm) {
            const auto thread_idx = itm.get_local_linear_id();
            const auto block_idx = itm.get_group_linear_id() + tmp_num_launched;
            const auto block_meta = mem_pool_write[0].deref_handle(
                block_group.block_descs)[block_idx];
            if (thread_idx == 0) {
              local_block[0] =
                  *mem_pool_write[0].deref_handle(block_meta.block);
              any_threads_pending[0] = false;
            }
            const auto *instructions_global_data =
                mem_pool_write[0].deref_handle(block_meta.instructions);
            auto instructions_for_block =
                local_instructions[itm.get_group_linear_id()];
            for (auto idx = thread_idx;
                 idx < block_meta.instructions.get_count();
                 idx += THREADS_PER_BLOCK) {
              instructions_for_block[idx] = instructions_global_data[idx];
            }
            itm.barrier(cl::sycl::access::fence_space::local_space);
            const RuntimeBlockType::Status status = local_block[0].evaluate(
                itm.get_group_linear_id(), thread_idx, itm, mem_pool_write,
                local_instructions, block_meta.instructions.get_count(),
                block_meta.num_threads,
                [indirect_call_handler_acc, mem_pool_write, indirect_all_acc](
                    const auto block, const auto funv, const auto tid,
                    const auto reg, const auto args) {
                  indirect_call_handler_acc[0].on_indirect_call(
                      indirect_all_acc, block, funv, tid, reg, args);
                },
                [indirect_call_handler_acc, mem_pool_write](const auto block) {
                  indirect_call_handler_acc[0].on_activate_block(mem_pool_write,
                                                                 block);
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
            if (!any_threads_pending[0] &&
                thread_idx < block_meta.num_threads) {
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
}
} // namespace FunGPU::EvaluatorV2
