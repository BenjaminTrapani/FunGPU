#pragma once

#include "core/evaluator_v2/lambda.hpp"
#include "core/evaluator_v2/program.hpp"
#include "core/evaluator_v2/runtime_value.hpp"
#include "core/portable_mem_pool.hpp"
#include "core/evaluator_v2/block_exec_group.hpp"

namespace FunGPU::EvaluatorV2 {
template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
class IndirectCallHandler {
public:
  // VM free parameters:
  // 1. Number of threads per block
  // 2. Number of registers per thread
  // 3. Maximum number of reactivations per pass
  // 4. Maximum number of indirect calls that may take place per pass
  // Derived parameters:
  // 1. Maximum number of blocks allocated per pass
  // 2. Max blocks scheduled per pass
  // Max blocks allocated per pass = ceil((Maximum number of indirect calls that
  // may take place per pass) / (Number of threads per block)) Max blocks
  // scheduled per pass = MaxNumReactivationsPerPass / Number of threads per
  // block
  //
  // When scheduling blocks, ensure that no more than
  // MAX_BLOCKS_SCHEDULED_PER_PASS are returned to the evaluator. Ensure that
  // the set of scheduled blocks will not exceed MaxNumIndirectCalls given the
  // number of indirect calls in each block's lambda. Preallocate space for
  // MAX_NUM_BLOCKS_ALLOCATED_PER_PASS blocks.
  static constexpr Index_t MAX_NUM_BLOCKS_ALLOCATED_PER_PASS =
      (MaxNumIndirectCalls + RuntimeBlockType::NumThreadsPerBlock - 1) /
      RuntimeBlockType::NumThreadsPerBlock;
  static constexpr Index_t MAX_BLOCKS_SCHEDULED_PER_PASS =
      std::min(MaxNumReactivations / RuntimeBlockType::NumThreadsPerBlock, MAX_NUM_BLOCKS_ALLOCATED_PER_PASS);

  // TODO captures and args are upper-bounded by the number of registers per
  // thread. Consider storing these in a dense array in IndirectCallRequest.
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
    Index_t num_indirect_calls_allocated = 0;
    std::array<Index_t, MAX_NUM_BLOCKS_ALLOCATED_PER_PASS> block_metadata_indexes;
  };

  class BlockReactivationRequestBuffer {
  public:
    void append(PortableMemPool::DeviceAccessor_t,
                PortableMemPool::Handle<RuntimeBlockType>);

    std::array<typename RuntimeBlockType::BlockMetadata, MaxNumReactivations>
        runtime_blocks_reactivated;
    Index_t num_runtime_blocks_reactivated = 0;
  };

  struct Buffers {
    Buffers(const std::size_t program_count)
        : max_num_threads_per_lambda(cl::sycl::range<1>(1)),
          num_blocks_scheduled(cl::sycl::range<1>(1)),
          max_num_instructions(cl::sycl::range<1>(1)),
          indirect_call_requests_by_block_data(
              std::shared_ptr<IndirectCallRequestBuffer[]>(
                  new IndirectCallRequestBuffer[program_count])),
          indirect_call_requests_by_block(
              indirect_call_requests_by_block_data.get(),
              cl::sycl::range<1>(program_count)),
          block_reactivation_requests_data(
              std::make_shared<BlockReactivationRequestBuffer>()),
          block_reactivation_requests_by_block(
              block_reactivation_requests_data.get(), cl::sycl::range<1>(1)),
          block_exec_group_data(std::shared_ptr<typename RuntimeBlockType::BlockMetadata[]>(new typename RuntimeBlockType::BlockMetadata[MAX_BLOCKS_SCHEDULED_PER_PASS])),
          block_exec_group(block_exec_group_data.get(), cl::sycl::range<1>(MAX_BLOCKS_SCHEDULED_PER_PASS)) {}

    using IndirectCallAccessorType =
        cl::sycl::accessor<IndirectCallRequestBuffer, 1,
                           cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>;
    using BlockReactivationRequestAccessorType =
        cl::sycl::accessor<BlockReactivationRequestBuffer, 1,
                           cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>;
    using MaxNumThreadsPerLambdaAccessorType =
        cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>;
    using NumBlocksScheduledAccessorType =
        cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>;
    using MaxNumInstructionsAccessorType =
        cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>;

    cl::sycl::buffer<Index_t> max_num_threads_per_lambda;
    cl::sycl::buffer<Index_t> num_blocks_scheduled;
    cl::sycl::buffer<Index_t> max_num_instructions;
    std::shared_ptr<IndirectCallRequestBuffer[]>
        indirect_call_requests_by_block_data;
    cl::sycl::buffer<IndirectCallRequestBuffer> indirect_call_requests_by_block;
    std::shared_ptr<BlockReactivationRequestBuffer>
        block_reactivation_requests_data;
    cl::sycl::buffer<BlockReactivationRequestBuffer>
        block_reactivation_requests_by_block;
    std::shared_ptr<typename RuntimeBlockType::BlockMetadata[]>
        block_exec_group_data;
    cl::sycl::buffer<typename RuntimeBlockType::BlockMetadata> block_exec_group;
  };

  static BlockExecGroup populate_block_exec_group(cl::sycl::queue &,
                          cl::sycl::buffer<PortableMemPool> &, Buffers &,
                          Program);

  static void on_indirect_call(typename Buffers::IndirectCallAccessorType,
                               PortableMemPool::Handle<RuntimeBlockType> caller,
                               FunctionValue, Index_t thread,
                               Index_t target_register,
                               PortableMemPool::ArrayHandle<RuntimeValue> args);
  static void
      on_activate_block(PortableMemPool::DeviceAccessor_t,
                        typename Buffers::BlockReactivationRequestAccessorType,
                        PortableMemPool::Handle<RuntimeBlockType>);
  static void
  reset_indirect_call_buffers(typename Buffers::IndirectCallAccessorType,
                              Index_t lambda_idx);
  static void reset_reactivations_buffer(
      typename Buffers::BlockReactivationRequestAccessorType);

  static Index_t num_blocks_for_lambda(
      Index_t lambda_idx,
      typename Buffers::IndirectCallAccessorType indirect_call_acc);

  static void
  setup_block_for_lambda(PortableMemPool::DeviceAccessor_t,
                         typename Buffers::IndirectCallAccessorType,
                         cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>,
                         typename Buffers::MaxNumThreadsPerLambdaAccessorType,
                         typename Buffers::NumBlocksScheduledAccessorType,
                         typename Buffers::MaxNumInstructionsAccessorType,
                         Index_t lambda_idx, Program);
  static void setup_block_for_reactivation(
      PortableMemPool::DeviceAccessor_t,
      typename Buffers::BlockReactivationRequestAccessorType,
      cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>,
      typename Buffers::NumBlocksScheduledAccessorType,
    typename Buffers::MaxNumInstructionsAccessorType);

  static void setup_block(PortableMemPool::DeviceAccessor_t,
                          typename Buffers::IndirectCallAccessorType,
                          cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> block_exec_acc, 
                          Index_t lambda_idx,
                          Index_t thread_idx);
};

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
Index_t IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                            MaxNumReactivations>::
    num_blocks_for_lambda(
        Index_t lambda_idx,
        typename Buffers::IndirectCallAccessorType indirect_call_acc) {
  return (indirect_call_acc[lambda_idx].num_indirect_call_reqs +
          RuntimeBlockType::NumThreadsPerBlock - 1) /
         RuntimeBlockType::NumThreadsPerBlock;
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void
IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                    MaxNumReactivations>::
    setup_block_for_lambda(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> block_exec_acc,
        const typename Buffers::MaxNumThreadsPerLambdaAccessorType max_num_threads_per_lambda,
        const typename Buffers::NumBlocksScheduledAccessorType num_blocks_scheduled_acc,
        const typename Buffers::MaxNumInstructionsAccessorType max_num_instructions_acc,
        const Index_t lambda_idx, const Program program) {
  auto &indirect_call_reqs = indirect_call_acc[lambda_idx];
  if (indirect_call_reqs.num_indirect_call_reqs == 0) {
    return;
  }
  const auto num_threads_per_block = RuntimeBlockType::NumThreadsPerBlock;
  const auto num_blocks_required =
      num_blocks_for_lambda(lambda_idx, indirect_call_acc);
  const auto &instructions_data =
      mem_pool_acc[0].deref_handle(program)[lambda_idx];
  Index_t max_num_instructions = 0;
  Index_t num_calls_remaining = indirect_call_reqs.num_indirect_call_reqs;
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      num_blocks_scheduled(num_blocks_scheduled_acc[0]);
  for (Index_t i = 0; i < num_blocks_required; ++i) {
    // TODO enforce total blocks scheduled limit and max total number of reactivations. Manage unscheduled calls.
    const auto num_threads =
        std::min(num_threads_per_block, num_calls_remaining);
    // TODO fan out pre allocated runtime value allocation across blocks in
    // separate kernel
    const auto maybe_pre_allocated_rvs =
        RuntimeBlockType::template pre_allocate_runtime_values<
            cl::sycl::access::target::device>(num_threads, mem_pool_acc,
                                              program, lambda_idx);
    if (!maybe_pre_allocated_rvs.has_value()) {
      break;
    }
    const auto runtime_block_handle = mem_pool_acc[0].alloc<RuntimeBlockType>(
        instructions_data.instructions, *maybe_pre_allocated_rvs, num_threads);
    if (runtime_block_handle == PortableMemPool::Handle<RuntimeBlockType>()) {
      for (Index_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        RuntimeBlockType::deallocate_runtime_values_for_thread(mem_pool_acc, *maybe_pre_allocated_rvs, thread_idx);
      }
      break;
    }
    const auto block_idx = num_blocks_scheduled.fetch_add(1U);
    block_exec_acc[block_idx] = typename RuntimeBlockType::BlockMetadata(
        runtime_block_handle, instructions_data.instructions, num_threads);
    max_num_instructions = std::max(max_num_instructions, instructions_data.instructions.get_count());
    num_calls_remaining -= num_threads;
    indirect_call_reqs.num_indirect_calls_allocated += num_threads;
    indirect_call_reqs.block_metadata_indexes[i] = block_idx;
  }
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      max_num_threads_per_lambda_atomic(max_num_threads_per_lambda[0]);
  max_num_threads_per_lambda_atomic.fetch_max(indirect_call_reqs.num_indirect_calls_allocated);
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
  max_num_instructions_atomic(max_num_instructions_acc[0]);
  max_num_instructions_atomic.fetch_max(max_num_instructions);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void
IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                    MaxNumReactivations>::
    setup_block_for_reactivation(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::BlockReactivationRequestAccessorType
            block_reactivation_requests_by_block,
        const cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> block_exec_acc,
        const typename Buffers::NumBlocksScheduledAccessorType num_blocks_scheduled_acc,
        const typename Buffers::MaxNumInstructionsAccessorType max_num_instructions_acc) {
  Index_t max_num_instructions = 0;
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      max_num_instructions_atomic(max_num_instructions_acc[0]);
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      num_blocks_scheduled(num_blocks_scheduled_acc[0]);
  for (Index_t i = 0; i < block_reactivation_requests_by_block[0]
              .num_runtime_blocks_reactivated; ++i) {
    block_exec_acc[num_blocks_scheduled.fetch_add(1U)] =
        block_reactivation_requests_by_block[0].runtime_blocks_reactivated[i];
    max_num_instructions_atomic.fetch_max(
        block_reactivation_requests_by_block[0].runtime_blocks_reactivated[i].instructions.get_count());
  }
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    setup_block(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> block_exec_acc, 
        const Index_t lambda_idx,
        const Index_t thread_idx) {
  const auto &indirect_call_reqs = indirect_call_acc[lambda_idx];
  if (thread_idx >= indirect_call_reqs.num_indirect_calls_allocated) {
    // TODO move unscheduled calls to a separate buffer
    return;
  }
  const auto &indirect_call_req =
      indirect_call_reqs.indirect_call_requests[thread_idx];
  const auto block_idx = indirect_call_reqs.block_metadata_indexes[thread_idx / RuntimeBlockType::NumThreadsPerBlock];
  const auto thread_idx_in_block =
      thread_idx % RuntimeBlockType::NumThreadsPerBlock;
  auto &target_block = *mem_pool_acc[0].deref_handle(block_exec_acc[block_idx].block);
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
  indirect_call_acc[lambda_idx].num_indirect_calls_allocated = 0;
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    reset_reactivations_buffer(
        typename Buffers::BlockReactivationRequestAccessorType
            block_reactivation_requests_by_block) {
  block_reactivation_requests_by_block[0].num_runtime_blocks_reactivated = 0;
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
                      typename Buffers::BlockReactivationRequestAccessorType
                          block_reactivation_requests_by_block,
                      const PortableMemPool::Handle<RuntimeBlockType> block) {
  block_reactivation_requests_by_block[0].append(mem_pool_acc, block);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
auto
IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                    MaxNumReactivations>::
    populate_block_exec_group(cl::sycl::queue &work_queue,
                            cl::sycl::buffer<PortableMemPool> &mem_pool_buffer,
                            Buffers &buffers, const Program program) -> BlockExecGroup {
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto max_num_threads_per_lambda_acc =
        buffers.max_num_threads_per_lambda
            .template get_access<cl::sycl::access::mode::write>(cgh);
    auto num_blocks_scheduled_acc =
        buffers.num_blocks_scheduled
            .template get_access<cl::sycl::access::mode::write>(cgh);
    auto max_num_instructions_acc =
        buffers.max_num_instructions
            .template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<class ResetIndirectCallHandlerBuffers>([=] {
      max_num_threads_per_lambda_acc[0] = 0;
      max_num_instructions_acc[0] = 0;
      num_blocks_scheduled_acc[0] = 0;
    });
  });

  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_exec_group_acc =
        buffers.block_exec_group
            .template get_access<cl::sycl::access::mode::write>(cgh);
    auto indirect_call_requests_by_block_acc =
        buffers.indirect_call_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_reactivation_requests_by_block_acc =
        buffers.block_reactivation_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto atomic_max_threads_per_lambda =
        buffers.max_num_threads_per_lambda
        .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto max_num_instructions_acc =
        buffers.max_num_instructions
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto num_blocks_scheduled_acc =
        buffers.num_blocks_scheduled
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class BlockGroupsPerLambda>(
        cl::sycl::range<1>(program.get_count() + 1),
        [mem_pool_write, block_exec_group_acc, program,
         indirect_call_requests_by_block_acc,
         block_reactivation_requests_by_block_acc, atomic_max_threads_per_lambda, max_num_instructions_acc, num_blocks_scheduled_acc](cl::sycl::item<1> itm) {
          const auto lambda_idx = itm.get_linear_id();
          if (lambda_idx < program.get_count()) {
            setup_block_for_lambda(mem_pool_write,
                                       indirect_call_requests_by_block_acc,
                                       block_exec_group_acc,
                                       atomic_max_threads_per_lambda,
                                       num_blocks_scheduled_acc,
                                       max_num_instructions_acc,
                                       lambda_idx, program);
          } else {
            setup_block_for_reactivation(
                mem_pool_write, block_reactivation_requests_by_block_acc,
                block_exec_group_acc, num_blocks_scheduled_acc,
                                       max_num_instructions_acc);
          }
        });
  });
  // Fill indirect call request blocks
  const auto max_num_threads_per_lambda_val =
      buffers.max_num_threads_per_lambda
          .template get_access<cl::sycl::access::mode::read>()[0];
  if (max_num_threads_per_lambda_val > 0) {
    work_queue.submit([&](cl::sycl::handler &cgh) {
        auto mem_pool_write =
            mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto block_exec_group_acc =
            buffers.block_exec_group
                .template get_access<cl::sycl::access::mode::read>(cgh);
        auto indirect_call_requests_by_block_acc =
            buffers.indirect_call_requests_by_block
                .template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class InitBlocksPerThread>(
            cl::sycl::range<2>(program.get_count(), max_num_threads_per_lambda_val),
            [mem_pool_write, block_exec_group_acc,
            indirect_call_requests_by_block_acc](const cl::sycl::item<2> itm) {
            setup_block(mem_pool_write, indirect_call_requests_by_block_acc,
                        block_exec_group_acc, itm.get_id(0),
                        itm.get_id(1));
            });
    });
  }

  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto indirect_call_requests_by_block_acc =
        buffers.indirect_call_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    auto reactivation_buffer_acc =
        buffers.block_reactivation_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class ResetBuffersAfterIndirectCallSchedule>(
        cl::sycl::range<1>(program.get_count() + 1),
        [mem_pool_acc, program,
         indirect_call_requests_by_block_acc,
         reactivation_buffer_acc](cl::sycl::item<1> itm) {
          const auto tid = static_cast<Index_t>(itm.get_linear_id());
          if (tid >= program.get_count()) {
            reset_reactivations_buffer(reactivation_buffer_acc);
          } else {
            reset_indirect_call_buffers(indirect_call_requests_by_block_acc,
                                        tid);
          }
        });
  });
  {
    auto num_blocks_scheduled_acc =
        buffers.num_blocks_scheduled
            .template get_access<cl::sycl::access::mode::read>();
    auto max_num_instructions_acc =
        buffers.max_num_instructions
            .template get_access<cl::sycl::access::mode::read>();
    return BlockExecGroup{.num_blocks = num_blocks_scheduled_acc[0], .max_num_instructions = max_num_instructions_acc[0]};
  }
}
} // namespace FunGPU::EvaluatorV2
