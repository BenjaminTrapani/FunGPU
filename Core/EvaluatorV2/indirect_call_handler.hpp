#pragma once

#include "Core/EvaluatorV2/lambda.hpp"
#include "Core/EvaluatorV2/program.hpp"
#include "Core/EvaluatorV2/runtime_block.hpp"
#include "Core/EvaluatorV2/runtime_value.hpp"
#include "Core/portable_mem_pool.hpp"

namespace FunGPU::EvaluatorV2 {
template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
class IndirectCallHandler {
public:
  using BlockExecGroupAccessor_t =
      cl::sycl::accessor<typename RuntimeBlockType::BlockExecGroup, 1,
                         cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;

  struct IndirectCallRequest {
    IndirectCallRequest(
        const PortableMemPool::Handle<RuntimeBlockType> caller,
        const PortableMemPool::ArrayHandle<RuntimeValue> captures,
        const Index_t calling_thread, const Index_t target_register,
        const PortableMemPool::ArrayHandle<RuntimeValue> args)
        : caller(caller), captures(captures), calling_thread(calling_thread),
          target_register(target_register), args(args) {}
    IndirectCallRequest() = default;

    PortableMemPool::Handle<RuntimeBlockType> caller;
    PortableMemPool::ArrayHandle<RuntimeValue> captures;
    Index_t calling_thread;
    Index_t target_register;
    PortableMemPool::ArrayHandle<RuntimeValue> args;
  };

  class IndirectCallRequestBuffer {
  public:
    void append(const IndirectCallRequest &);

    std::array<IndirectCallRequest, MaxNumIndirectCalls> indirect_call_requests;
    Index_t num_indirect_call_reqs = 0;
  };

  struct Buffers {
    Buffers(const std::size_t program_count)
        : block_exec_group_per_lambda(cl::sycl::range<1>(program_count + 1)),
          max_num_threads_per_lambda(cl::sycl::range<1>(1)),
          total_num_blocks(cl::sycl::range<1>(1)),
          result(cl::sycl::range<1>(1)), copy_begin_idx(cl::sycl::range<1>(1)),
          indirect_call_requests_by_block(cl::sycl::range<1>(program_count)) {}

    void update_for_num_lambdas(const std::size_t program_count) {
      indirect_call_requests_by_block_data =
          std::shared_ptr<IndirectCallRequestBuffer[]>(
              new IndirectCallRequestBuffer[program_count]);
      indirect_call_requests_by_block =
          cl::sycl::buffer<IndirectCallRequestBuffer>(
              indirect_call_requests_by_block_data.get(),
              cl::sycl::range<1>(program_count));
    }

    using IndirectCallAccessorType =
        cl::sycl::accessor<IndirectCallRequestBuffer, 1,
                           cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>;

    cl::sycl::buffer<typename RuntimeBlockType::BlockExecGroup>
        block_exec_group_per_lambda;
    cl::sycl::buffer<Index_t> max_num_threads_per_lambda;
    cl::sycl::buffer<Index_t> total_num_blocks;
    cl::sycl::buffer<typename RuntimeBlockType::BlockExecGroup> result;
    cl::sycl::buffer<Index_t> copy_begin_idx;
    std::shared_ptr<IndirectCallRequestBuffer[]>
        indirect_call_requests_by_block_data;
    cl::sycl::buffer<IndirectCallRequestBuffer> indirect_call_requests_by_block;
  };

  class BlockReactivationRequestBuffer {
  public:
    void append(PortableMemPool::DeviceAccessor_t,
                PortableMemPool::Handle<RuntimeBlockType>);

    std::array<typename RuntimeBlockType::BlockMetadata, MaxNumReactivations>
        runtime_blocks_reactivated;
    Index_t num_runtime_blocks_reactivated = 0;
  };

  static typename RuntimeBlockType::BlockExecGroup create_block_exec_group(
      cl::sycl::queue &, cl::sycl::buffer<PortableMemPool> &,
      cl::sycl::buffer<IndirectCallHandler> &, Buffers &, Program);

  void on_indirect_call(typename Buffers::IndirectCallAccessorType,
                        PortableMemPool::Handle<RuntimeBlockType> caller,
                        FunctionValue, Index_t thread, Index_t target_register,
                        PortableMemPool::ArrayHandle<RuntimeValue> args);
  void on_activate_block(PortableMemPool::DeviceAccessor_t,
                         PortableMemPool::Handle<RuntimeBlockType>);
  void reset_indirect_call_buffers(typename Buffers::IndirectCallAccessorType,
                                   Index_t lambda_idx);
  void reset_reactivations_buffer();

  typename RuntimeBlockType::BlockExecGroup
  setup_block_for_lambda(PortableMemPool::DeviceAccessor_t,
                         typename Buffers::IndirectCallAccessorType,
                         Index_t lambda_idx, Program) const;
  typename RuntimeBlockType::BlockExecGroup
      setup_block_for_reactivation(PortableMemPool::DeviceAccessor_t) const;

  void setup_block(PortableMemPool::DeviceAccessor_t,
                   typename Buffers::IndirectCallAccessorType,
                   BlockExecGroupAccessor_t, Index_t lambda_idx,
                   Index_t thread_idx) const;

  BlockReactivationRequestBuffer block_reactivation_requests_by_block;
};

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
typename RuntimeBlockType::BlockExecGroup
IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                    MaxNumReactivations>::
    setup_block_for_lambda(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const Index_t lambda_idx, const Program program) const {
  const auto &indirect_call_reqs = indirect_call_acc[lambda_idx];
  if (indirect_call_reqs.num_indirect_call_reqs == 0) {
    return typename RuntimeBlockType::BlockExecGroup(
        PortableMemPool::ArrayHandle<
            typename RuntimeBlockType::BlockMetadata>(),
        0);
  }
  const auto num_threads_per_block = RuntimeBlockType::NumThreadsPerBlock;
  const auto num_blocks_required =
      (indirect_call_reqs.num_indirect_call_reqs + num_threads_per_block - 1) /
      num_threads_per_block;
  const auto block_meta_handle =
      mem_pool_acc[0].alloc_array<typename RuntimeBlockType::BlockMetadata>(
          num_blocks_required);
  const auto &instructions_data =
      mem_pool_acc[0].deref_handle(program)[lambda_idx];
  auto *block_metadata = mem_pool_acc[0].deref_handle(block_meta_handle);
  Index_t max_num_instructions = 0;
  Index_t num_calls_remaining = indirect_call_reqs.num_indirect_call_reqs;
  for (Index_t i = 0; i < num_blocks_required; ++i) {
    const auto num_threads =
        std::min(num_threads_per_block, num_calls_remaining);
    // TODO fan out pre allocated runtime value allocation across blocks in
    // separate kernel
    const auto pre_allocated_rvs =
        RuntimeBlockType::template pre_allocate_runtime_values<
            cl::sycl::access::target::device>(num_threads, mem_pool_acc,
                                              program, lambda_idx);
    const auto runtime_block_handle = mem_pool_acc[0].alloc<RuntimeBlockType>(
        instructions_data.instructions, pre_allocated_rvs, num_threads);
    block_metadata[i] = typename RuntimeBlockType::BlockMetadata(
        runtime_block_handle, instructions_data.instructions, num_threads);
    max_num_instructions = std::max(max_num_instructions,
                                    instructions_data.instructions.get_count());
    num_calls_remaining -= num_threads;
  }
  return typename RuntimeBlockType::BlockExecGroup(block_meta_handle,
                                                   max_num_instructions);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
typename RuntimeBlockType::BlockExecGroup
IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                    MaxNumReactivations>::
    setup_block_for_reactivation(
        PortableMemPool::DeviceAccessor_t mem_pool_acc) const {
  const auto block_meta_handle =
      mem_pool_acc[0].alloc_array<typename RuntimeBlockType::BlockMetadata>(
          block_reactivation_requests_by_block.num_runtime_blocks_reactivated);
  Index_t max_num_instructions = 0;
  auto *block_meta_data = mem_pool_acc[0].deref_handle(block_meta_handle);
  for (Index_t i = 0; i < block_meta_handle.get_count(); ++i) {
    block_meta_data[i] =
        block_reactivation_requests_by_block.runtime_blocks_reactivated[i];
    max_num_instructions = std::max(
        max_num_instructions, block_meta_data[i].instructions.get_count());
  }
  return typename RuntimeBlockType::BlockExecGroup(block_meta_handle,
                                                   max_num_instructions);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    setup_block(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        BlockExecGroupAccessor_t block_exec_acc, const Index_t lambda_idx,
        const Index_t thread_idx) const {
  const auto &indirect_call_reqs = indirect_call_acc[lambda_idx];
  if (thread_idx >= indirect_call_reqs.num_indirect_call_reqs) {
    return;
  }
  const auto &indirect_call_req =
      indirect_call_reqs.indirect_call_requests[thread_idx];
  const auto block_idx = thread_idx / RuntimeBlockType::NumThreadsPerBlock;
  const auto thread_idx_in_block =
      thread_idx % RuntimeBlockType::NumThreadsPerBlock;
  auto &target_block = *mem_pool_acc[0].deref_handle(
      mem_pool_acc[0]
          .deref_handle(block_exec_acc[lambda_idx].block_descs)[block_idx]
          .block);
  // captures first, then args.
  auto &register_set = target_block.registers[thread_idx_in_block];
  const auto *capture_data =
      mem_pool_acc[0].deref_handle(indirect_call_req.captures);
  auto it = std::copy(capture_data,
                      capture_data + indirect_call_req.captures.get_count(),
                      register_set.begin());
  const auto *arg_data = mem_pool_acc[0].deref_handle(indirect_call_req.args);
  std::copy(arg_data, arg_data + indirect_call_req.args.get_count(), it);
  mem_pool_acc[0].dealloc_array(indirect_call_req.args);

  auto &target_address_data = target_block.target_data[thread_idx_in_block];
  target_address_data.block = indirect_call_req.caller;
  target_address_data.thread = indirect_call_req.calling_thread;
  target_address_data.register_idx = indirect_call_req.target_register;
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::IndirectCallRequestBuffer::
    append(const IndirectCallRequest &req) {
  cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      count(num_indirect_call_reqs);
  const auto target_idx = count.fetch_add(1U);
  indirect_call_requests[target_idx] = req;
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::BlockReactivationRequestBuffer::
    append(PortableMemPool::DeviceAccessor_t mem_pool_acc,
           const PortableMemPool::Handle<RuntimeBlockType> block) {
  cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      count(num_runtime_blocks_reactivated);
  const auto target_idx = count.fetch_add(1U);
  runtime_blocks_reactivated[target_idx] =
      mem_pool_acc[0].deref_handle(block)->block_metadata();
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    reset_indirect_call_buffers(
        typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const Index_t lambda_idx) {
  indirect_call_acc[lambda_idx].num_indirect_call_reqs = 0;
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::reset_reactivations_buffer() {
  block_reactivation_requests_by_block.num_runtime_blocks_reactivated = 0;
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    on_indirect_call(
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const PortableMemPool::Handle<RuntimeBlockType> caller,
        const FunctionValue funv, const Index_t thread,
        const Index_t target_register,
        const PortableMemPool::ArrayHandle<RuntimeValue> args) {
  const auto target_block_idx = funv.block_idx;
  const IndirectCallRequest ind_call_req(caller, funv.captures.unpack(), thread,
                                         target_register, args);
  auto &indirect_call_req_buffer_for_block =
      indirect_call_acc[target_block_idx];
  indirect_call_req_buffer_for_block.append(ind_call_req);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    on_activate_block(PortableMemPool::DeviceAccessor_t mem_pool_acc,
                      const PortableMemPool::Handle<RuntimeBlockType> block) {
  block_reactivation_requests_by_block.append(mem_pool_acc, block);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
typename RuntimeBlockType::BlockExecGroup
IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                    MaxNumReactivations>::
    create_block_exec_group(
        cl::sycl::queue &work_queue,
        cl::sycl::buffer<PortableMemPool> &mem_pool_buffer,
        cl::sycl::buffer<IndirectCallHandler> &indirect_call_handler,
        Buffers &buffers, const Program program) {
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto max_num_threads_per_lambda_acc =
        buffers.max_num_threads_per_lambda
            .template get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto total_num_blocks_acc =
        buffers.total_num_blocks
            .template get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto copy_begin_idx_acc =
        buffers.copy_begin_idx
            .template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class ResetIndirectCallHandlerBuffers>([=] {
      max_num_threads_per_lambda_acc[0] = 0;
      total_num_blocks_acc[0] = 0;
      copy_begin_idx_acc[0] = 0;
    });
  });

  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_exec_group_per_lambda_acc =
        buffers.block_exec_group_per_lambda
            .template get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto indirect_call_handler_acc =
        indirect_call_handler
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto atomic_max_threads_per_lambda =
        buffers.max_num_threads_per_lambda
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto total_num_blocks_acc =
        buffers.total_num_blocks
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto indirect_call_requests_by_block_acc =
        buffers.indirect_call_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class BlockGroupsPerLambda>(
        cl::sycl::range<1>(program.get_count() + 1),
        [mem_pool_write, block_exec_group_per_lambda_acc,
         indirect_call_handler_acc, program, atomic_max_threads_per_lambda,
         total_num_blocks_acc,
         indirect_call_requests_by_block_acc](cl::sycl::item<1> itm) {
          const auto lambda_idx = itm.get_linear_id();
          if (lambda_idx < program.get_count()) {
            block_exec_group_per_lambda_acc[lambda_idx] =
                indirect_call_handler_acc[0].setup_block_for_lambda(
                    mem_pool_write, indirect_call_requests_by_block_acc,
                    lambda_idx, program);
          } else {
            block_exec_group_per_lambda_acc[lambda_idx] =
                indirect_call_handler_acc[0].setup_block_for_reactivation(
                    mem_pool_write);
          }
          cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                               cl::sycl::memory_scope::device,
                               cl::sycl::access::address_space::global_space>
              max_num_threads_per_lambda(atomic_max_threads_per_lambda[0]);
          const auto num_blocks_in_lambda =
              block_exec_group_per_lambda_acc[lambda_idx]
                  .block_descs.get_count();
          max_num_threads_per_lambda.fetch_max(
              num_blocks_in_lambda * RuntimeBlockType::NumThreadsPerBlock);
          cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                               cl::sycl::memory_scope::device,
                               cl::sycl::access::address_space::global_space>
              total_num_blocks_atomic(total_num_blocks_acc[0]);
          total_num_blocks_atomic.fetch_add(num_blocks_in_lambda);
        });
  });
  // Fill indirect call request blocks
  const auto max_num_threads_per_lambda_val =
      buffers.max_num_threads_per_lambda
          .template get_access<cl::sycl::access::mode::read>()[0];
  if (max_num_threads_per_lambda_val == 0) {
    throw std::invalid_argument("No threads across any lambdas");
  }
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_exec_group_per_lambda_acc =
        buffers.block_exec_group_per_lambda
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto indirect_call_handler_acc =
        indirect_call_handler
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto indirect_call_requests_by_block_acc =
        buffers.indirect_call_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class InitBlocksPerThread>(
        cl::sycl::range<2>(program.get_count(), max_num_threads_per_lambda_val),
        [mem_pool_write, block_exec_group_per_lambda_acc,
         indirect_call_handler_acc,
         indirect_call_requests_by_block_acc](const cl::sycl::item<2> itm) {
          indirect_call_handler_acc[0].setup_block(
              mem_pool_write, indirect_call_requests_by_block_acc,
              block_exec_group_per_lambda_acc, itm.get_id(0), itm.get_id(1));
        });
  });

  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto total_num_blocks_acc =
        buffers.total_num_blocks
            .template get_access<cl::sycl::access::mode::read>(cgh);
    auto result_acc =
        buffers.result
            .template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class InitResult>([mem_pool_write, total_num_blocks_acc,
                                       result_acc] {
      const auto all_block_descs =
          mem_pool_write[0]
              .template alloc_array<typename RuntimeBlockType::BlockMetadata>(
                  total_num_blocks_acc[0]);
      result_acc[0] =
          typename RuntimeBlockType::BlockExecGroup(all_block_descs, 0);
    });
  });

  // Merge block exec groups into one.
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto result_acc =
        buffers.result.template get_access<cl::sycl::access::mode::read_write>(
            cgh);
    auto block_exec_group_per_lambda_acc =
        buffers.block_exec_group_per_lambda
            .template get_access<cl::sycl::access::mode::read>(cgh);
    auto copy_begin_atomic_acc =
        buffers.copy_begin_idx
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class MergeIntoResult>(
        cl::sycl::range<2>(program.get_count() + 1,
                           (max_num_threads_per_lambda_val +
                            RuntimeBlockType::NumThreadsPerBlock - 1) /
                               RuntimeBlockType::NumThreadsPerBlock),
        [mem_pool_write, result_acc, block_exec_group_per_lambda_acc,
         copy_begin_atomic_acc](cl::sycl::item<2> itm) {
          auto *target_block_descs =
              mem_pool_write[0].deref_handle(result_acc[0].block_descs);
          cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                               cl::sycl::memory_scope::device,
                               cl::sycl::access::address_space::global_space>
              copy_begin_atomic(copy_begin_atomic_acc[0]);
          const auto source_block_descs =
              block_exec_group_per_lambda_acc[itm.get_id(0)].block_descs;
          if (source_block_descs ==
              PortableMemPool::ArrayHandle<
                  typename RuntimeBlockType::BlockMetadata>()) {
            return;
          }
          const auto *source_data =
              mem_pool_write[0].deref_handle(source_block_descs);
          if (itm.get_id(1) < source_block_descs.get_count()) {
            target_block_descs[copy_begin_atomic.fetch_add(1U)] =
                source_data[itm.get_id(1)];
          }
          if (itm.get_id(1) == 0) {
            cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                                 cl::sycl::memory_scope::device,
                                 cl::sycl::access::address_space::global_space>
                max_num_instructions_atomic(result_acc[0].max_num_instructions);
            max_num_instructions_atomic.fetch_max(
                block_exec_group_per_lambda_acc[itm.get_id(0)]
                    .max_num_instructions);
          }
        });
  });

  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto indirect_call_handler_acc =
        indirect_call_handler
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_exec_group_per_lambda_acc =
        buffers.block_exec_group_per_lambda
            .template get_access<cl::sycl::access::mode::read>(cgh);
    auto indirect_call_requests_by_block_acc =
        buffers.indirect_call_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class ResetBuffersAfterIndirectCallSchedule>(
        cl::sycl::range<1>(program.get_count() + 1),
        [indirect_call_handler_acc, mem_pool_acc, program,
         block_exec_group_per_lambda_acc,
         indirect_call_requests_by_block_acc](cl::sycl::item<1> itm) {
          const auto tid = static_cast<Index_t>(itm.get_linear_id());
          if (tid >= program.get_count()) {
            indirect_call_handler_acc[0].reset_reactivations_buffer();
          } else {
            indirect_call_handler_acc[0].reset_indirect_call_buffers(
                indirect_call_requests_by_block_acc, tid);
          }
          const auto source_block_descs =
              block_exec_group_per_lambda_acc[itm.get_linear_id()].block_descs;
          mem_pool_acc[0].dealloc_array(source_block_descs);
        });
  });

  return buffers.result.template get_access<cl::sycl::access::mode::read>()[0];
}
} // namespace FunGPU::EvaluatorV2
