#include "Core/EvaluatorV2/Evaluator.hpp"
#include "Core/EvaluatorV2/RuntimeBlock.hpp"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/PortableMemPool.hpp"
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
Evaluator::Evaluator(cl::sycl::buffer<PortableMemPool> buffer)
    : mem_pool_buffer_(buffer) {
  std::cout << "Running on "
            << work_queue_.get_device().get_info<cl::sycl::info::device::name>()
            << ", block size: " << sizeof(RuntimeBlockType) << std::endl;
}

RuntimeValue Evaluator::compute(const Program program) {
  first_block_ = construct_initial_block(program);
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto indirect_call_acc =
        indirect_call_handler_buffer_
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    auto mem_pool_acc =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    const auto first_block_tmp = first_block_;
    cgh.single_task(
        [indirect_call_acc, mem_pool_acc, first_block_tmp, program] {
          indirect_call_acc[0].update_for_num_lambdas(mem_pool_acc,
                                                      program.GetCount());
          FunctionValue funv;
          funv.block_idx = 0;
          funv.captures = PortableMemPool::ArrayHandle<RuntimeValue>();
          indirect_call_acc[0].on_indirect_call(
              mem_pool_acc, first_block_tmp, funv, 0, 0,
              PortableMemPool::ArrayHandle<RuntimeValue>());
        });
  });

  while (true) {
    std::cout << "Scheduling next batch" << std::endl;
    const auto maybe_next_batch = schedule_next_batch(program);
    if (!maybe_next_batch.has_value()) {
      break;
    }
    std::cout << "Running eval step" << std::endl;
    run_eval_step(*maybe_next_batch);
  }

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
      const auto &main_lambda = mem_pool_acc[0].derefHandle(program)[0];
      const auto block =
          mem_pool_acc[0].Alloc<RuntimeBlockType>(main_lambda.instructions, 1);
      result_acc[0] = block;
      auto &block_data = *mem_pool_acc[0].derefHandle(block);
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
      const auto &first_block_data = *mem_pool_acc[0].derefHandle(first_block);
      result_acc[0] = first_block_data.registers[0][0];
    });
  });
  return result.get_access<cl::sycl::access::mode::read>()[0];
}

auto Evaluator::schedule_next_batch(const Program program)
    -> std::optional<RuntimeBlockType::BlockExecGroup> {
  const auto next_batch = IndirectCallHandlerType::create_block_exec_group(
      work_queue_, mem_pool_buffer_, indirect_call_handler_buffer_, program);
  if (next_batch.block_descs.GetCount() > 1) {
    /*std::cout << "Multiple blocks returned" << std::endl;
    auto mem_pool_acc = mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>();
    const auto* block_metadata = mem_pool_acc[0].derefHandle(next_batch.block_descs);
    for (Index_t i = 0; i < next_batch.block_descs.GetCount(); ++i) {
      Index_t lambda_idx = std::numeric_limits<Index_t>::max();
      const auto* program_data = mem_pool_acc[0].derefHandle(program);
      for (Index_t j = 0; j < program.GetCount(); ++j) {
        if (program_data[j].instructions == block_metadata[i].instructions) {
          lambda_idx = j;
          break;
        }
      }
      std::cout << " block: num threads " << block_metadata[i].num_threads << " for lambda " << lambda_idx << std::endl;
    }*/
    return next_batch;
  }
  if (next_batch.block_descs.GetCount() != 1) {
    throw std::invalid_argument("Expected at least one block");
  }
  cl::sycl::buffer<bool> is_initial_block_ready_again(cl::sycl::range<1>(1));
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto result_acc =
        is_initial_block_ready_again
            .get_access<cl::sycl::access::mode::discard_write>(cgh);
    const auto first_block_tmp = first_block_;
    cgh.single_task([mem_pool_acc, next_batch, is_initial_block_ready_again,
                     first_block_tmp, result_acc] {
      const auto &block_meta =
          mem_pool_acc[0].derefHandle((next_batch.block_descs))[0];
      result_acc[0] = block_meta.block == first_block_tmp;
    });
  });
  if (is_initial_block_ready_again
          .get_access<cl::sycl::access::mode::read>()[0]) {
    return std::nullopt;
  }
  return next_batch;
}

void Evaluator::run_eval_step(
    const RuntimeBlockType::BlockExecGroup block_group) {
  work_queue_.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto indirect_call_handler_acc =
        indirect_call_handler_buffer_
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cl::sycl::accessor<RuntimeBlockType, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        local_block(cl::sycl::range<1>(1), cgh);
    RuntimeBlockType::InstructionLocalMemAccessor local_instructions(
        cl::sycl::range<2>(block_group.block_descs.GetCount(),
                           block_group.max_num_instructions),
        cgh);
    cgh.parallel_for<class TestEvalLoop>(
        cl::sycl::nd_range<1>(THREADS_PER_BLOCK *
                                  block_group.block_descs.GetCount(),
                              THREADS_PER_BLOCK),
        [mem_pool_write, block_group, local_block, local_instructions,
         indirect_call_handler_acc](cl::sycl::nd_item<1> itm) {
          const auto thread_idx = itm.get_local_linear_id();
          const auto block_idx = itm.get_group_linear_id();
          const auto block_meta =
              mem_pool_write[0].derefHandle(block_group.block_descs)[block_idx];
          if (thread_idx == 0) {
            local_block[0] = *mem_pool_write[0].derefHandle(block_meta.block);
          }
          const auto *instructions_global_data =
              mem_pool_write[0].derefHandle(block_meta.instructions);
          auto instructions_for_block = local_instructions[block_idx];
          for (auto idx = thread_idx; idx < block_meta.instructions.GetCount();
               idx += THREADS_PER_BLOCK) {
            instructions_for_block[idx] = instructions_global_data[idx];
          }
          itm.barrier();
          if (thread_idx >= block_meta.num_threads) {
            return;
          }
          RuntimeBlockType::Status status = local_block[0].evaluate(
              block_idx, thread_idx, mem_pool_write, local_instructions,
              block_meta.instructions.GetCount(),
              [indirect_call_handler_acc, mem_pool_write](
                  const auto block, const auto funv, const auto tid,
                  const auto reg, const auto args) {
                indirect_call_handler_acc[0].on_indirect_call(
                    mem_pool_write, block, funv, tid, reg, args);
              },
              [indirect_call_handler_acc, mem_pool_write](const auto block) {
                indirect_call_handler_acc[0].on_activate_block(mem_pool_write,
                                                               block);
              });
          // TODO deallocate block if all complete in thread.
          if (thread_idx == 0) {
            *mem_pool_write[0].derefHandle(block_meta.block) = local_block[0];
          }
        });
  });
}
} // namespace FunGPU::EvaluatorV2
