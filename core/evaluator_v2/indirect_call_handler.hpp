#pragma once

#include "core/evaluator_v2/block_exec_group.hpp"
#include "core/evaluator_v2/lambda.hpp"
#include "core/evaluator_v2/program.hpp"
#include "core/evaluator_v2/runtime_value.hpp"
#include "core/portable_mem_pool.hpp"

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
  // Max blocks allocated per pass =(Maximum number of indirect calls that
  // may take place per pass
  // Max blocks scheduled per pass = MaxNumReactivationsPerPass
  //
  // When scheduling blocks, ensure that no more than
  // MAX_BLOCKS_SCHEDULED_PER_PASS are returned to the evaluator. Ensure that
  // the set of scheduled blocks will not exceed MaxNumIndirectCalls given the
  // number of indirect calls in each block's lambda. Preallocate space for
  // MAX_NUM_BLOCKS_ALLOCATED_PER_PASS blocks.

  // TODO: use per-lambda indirect call count limit during scheduling to allow
  // for smaller values of MaxNumIndirectCalls.
  // MAX_NUM_BLOCKS_ALLOCATED_PER_PASS is then MaxNumIndirectCalls *
  // num_lambdas, which is not available at compile-time.

  static constexpr Index_t MAX_NUM_INDIRECT_CALLS = MaxNumIndirectCalls;
  static constexpr Index_t MAX_NUM_BLOCKS_ALLOCATED_PER_PASS =
      MaxNumIndirectCalls;
  static constexpr Index_t MAX_BLOCKS_SCHEDULED_PER_PASS =
      std::min(MaxNumReactivations, MAX_NUM_BLOCKS_ALLOCATED_PER_PASS);

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
    std::array<Index_t, MAX_NUM_BLOCKS_ALLOCATED_PER_PASS>
        block_metadata_indexes;
  };

  class BlockReactivationRequestBuffer {
  public:
    void append(PortableMemPool::DeviceAccessor_t,
                const PortableMemPool::Handle<RuntimeBlockType>);
    void append(const typename RuntimeBlockType::BlockMetadata &);

    std::array<typename RuntimeBlockType::BlockMetadata, MaxNumReactivations>
        runtime_blocks_reactivated;
    Index_t num_runtime_blocks_reactivated = 0;
    Index_t num_reactivations_scheduled = 0;
  };

  struct Buffers {
    Buffers(const std::size_t program_count)
        : max_num_threads_per_lambda(cl::sycl::range<1>(1)),
          num_blocks_scheduled(cl::sycl::range<1>(1)),
          max_num_instructions(cl::sycl::range<1>(1)),
          total_indirect_calls_across_scheduled_blocks(cl::sycl::range<1>(1)),
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
          block_exec_group_data(
              std::shared_ptr<typename RuntimeBlockType::BlockMetadata[]>(
                  new typename RuntimeBlockType::BlockMetadata
                      [MAX_BLOCKS_SCHEDULED_PER_PASS])),
          block_exec_group(block_exec_group_data.get(),
                           cl::sycl::range<1>(MAX_BLOCKS_SCHEDULED_PER_PASS)),
          buffered_indirect_call_requests(program_count) {}

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
    cl::sycl::buffer<Index_t> total_indirect_calls_across_scheduled_blocks;
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
    std::vector<std::deque<IndirectCallRequest>>
        buffered_indirect_call_requests;
    std::deque<typename RuntimeBlockType::BlockMetadata> buffered_reactivations;
  };

  static BlockExecGroup
  populate_block_exec_group(cl::sycl::queue &,
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

  static void setup_block_for_lambda(
      PortableMemPool::DeviceAccessor_t,
      typename Buffers::IndirectCallAccessorType,
      cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1,
                         cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>,
      typename Buffers::MaxNumThreadsPerLambdaAccessorType,
      typename Buffers::NumBlocksScheduledAccessorType,
      typename Buffers::MaxNumInstructionsAccessorType, Index_t lambda_idx,
      Program,
      cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          total_indirect_calls_across_scheduled_blocks_acc);
  static void setup_block_for_reactivation(
      PortableMemPool::DeviceAccessor_t,
      typename Buffers::BlockReactivationRequestAccessorType,
      cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1,
                         cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>,
      typename Buffers::NumBlocksScheduledAccessorType,
      typename Buffers::MaxNumInstructionsAccessorType,
      cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          total_indirect_calls_across_scheduled_blocks_acc);

  static void
  setup_block(PortableMemPool::DeviceAccessor_t,
              typename Buffers::IndirectCallAccessorType,
              cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1,
                                 cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer>
                  block_exec_acc,
              Index_t lambda_idx, Index_t thread_idx);
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
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    setup_block_for_lambda(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1,
                                 cl::sycl::access::mode::write,
                                 cl::sycl::access::target::global_buffer>
            block_exec_acc,
        const typename Buffers::MaxNumThreadsPerLambdaAccessorType
            max_num_threads_per_lambda,
        const typename Buffers::NumBlocksScheduledAccessorType
            num_blocks_scheduled_acc,
        const typename Buffers::MaxNumInstructionsAccessorType
            max_num_instructions_acc,
        const Index_t lambda_idx, const Program program,
        const cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::global_buffer>
            total_indirect_calls_across_scheduled_blocks_acc) {
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
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                             cl::sycl::memory_scope::device,
                             cl::sycl::access::address_space::global_space>
      total_indirect_calls_across_scheduled_blocks(
          total_indirect_calls_across_scheduled_blocks_acc[0]);
  for (Index_t i = 0; i < num_blocks_required; ++i) {
    // TODO enforce total blocks scheduled limit and max total number of
    // reactivations. Manage unscheduled calls.
    const auto num_threads =
        std::min(num_threads_per_block, num_calls_remaining);
    if (const auto total_calls =
            total_indirect_calls_across_scheduled_blocks.fetch_add(
                instructions_data.instruction_properties
                    .total_num_indirect_calls);
        total_calls +
            instructions_data.instruction_properties.total_num_indirect_calls >
        MaxNumIndirectCalls) {
      total_indirect_calls_across_scheduled_blocks.fetch_sub(
          instructions_data.instruction_properties.total_num_indirect_calls);
      break;
    }
    // TODO fan out pre allocated runtime value allocation across blocks in
    // separate kernel
    const auto maybe_pre_allocated_rvs =
        RuntimeBlockType::template pre_allocate_runtime_values<
            cl::sycl::access::target::device>(num_threads, mem_pool_acc,
                                              program, lambda_idx);
    if (!maybe_pre_allocated_rvs.has_value()) {
      break;
    }
    const auto lambda_handle = program.element_handle(lambda_idx);
    const auto runtime_block_handle = mem_pool_acc[0].alloc<RuntimeBlockType>(
        lambda_handle, *maybe_pre_allocated_rvs, num_threads);
    const auto dealloc_all_runtime_values = [&]() {
      for (Index_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        RuntimeBlockType::deallocate_runtime_values_for_thread(
            mem_pool_acc, *maybe_pre_allocated_rvs, thread_idx);
      }
    };
    if (runtime_block_handle == PortableMemPool::Handle<RuntimeBlockType>()) {
      dealloc_all_runtime_values();
      break;
    }
    const auto block_idx = num_blocks_scheduled.fetch_add(1U);
    if (block_idx >= MAX_BLOCKS_SCHEDULED_PER_PASS) {
      mem_pool_acc[0].dealloc(runtime_block_handle);
      dealloc_all_runtime_values();
      break;
    }
    block_exec_acc[block_idx] = typename RuntimeBlockType::BlockMetadata(
        runtime_block_handle, lambda_handle, num_threads);
    max_num_instructions = std::max(max_num_instructions,
                                    instructions_data.instructions.get_count());
    num_calls_remaining -= num_threads;
    indirect_call_reqs.num_indirect_calls_allocated += num_threads;
    indirect_call_reqs.block_metadata_indexes[i] = block_idx;
  }
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                             cl::sycl::memory_scope::device,
                             cl::sycl::access::address_space::global_space>
      max_num_threads_per_lambda_atomic(max_num_threads_per_lambda[0]);
  max_num_threads_per_lambda_atomic.fetch_max(
      indirect_call_reqs.num_indirect_calls_allocated);
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                             cl::sycl::memory_scope::device,
                             cl::sycl::access::address_space::global_space>
      max_num_instructions_atomic(max_num_instructions_acc[0]);
  max_num_instructions_atomic.fetch_max(max_num_instructions);
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    setup_block_for_reactivation(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::BlockReactivationRequestAccessorType
            block_reactivation_requests_by_block,
        const cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1,
                                 cl::sycl::access::mode::write,
                                 cl::sycl::access::target::global_buffer>
            block_exec_acc,
        const typename Buffers::NumBlocksScheduledAccessorType
            num_blocks_scheduled_acc,
        const typename Buffers::MaxNumInstructionsAccessorType
            max_num_instructions_acc,
        const cl::sycl::accessor<Index_t, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::global_buffer>
            total_indirect_calls_across_scheduled_blocks_acc) {
  Index_t max_num_instructions = 0;
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                             cl::sycl::memory_scope::device,
                             cl::sycl::access::address_space::global_space>
      max_num_instructions_atomic(max_num_instructions_acc[0]);
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                             cl::sycl::memory_scope::device,
                             cl::sycl::access::address_space::global_space>
      total_indirect_calls_across_scheduled_blocks(
          total_indirect_calls_across_scheduled_blocks_acc[0]);
  const cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                             cl::sycl::memory_scope::device,
                             cl::sycl::access::address_space::global_space>
      num_blocks_scheduled(num_blocks_scheduled_acc[0]);
  for (Index_t i = 0;
       i <
       block_reactivation_requests_by_block[0].num_runtime_blocks_reactivated;
       ++i) {
    const auto total_num_indirect_calls =
        mem_pool_acc[0]
            .deref_handle(block_reactivation_requests_by_block[0]
                              .runtime_blocks_reactivated[i]
                              .lambda)
            ->instruction_properties.total_num_indirect_calls;
    if (const auto total_calls =
            total_indirect_calls_across_scheduled_blocks.fetch_add(
                total_num_indirect_calls);
        total_calls + total_num_indirect_calls > MaxNumIndirectCalls) {
      total_indirect_calls_across_scheduled_blocks.fetch_sub(
          total_num_indirect_calls);
      break;
    }

    const auto block_idx = num_blocks_scheduled.fetch_add(1U);
    if (block_idx >= MAX_BLOCKS_SCHEDULED_PER_PASS) {
      total_indirect_calls_across_scheduled_blocks.fetch_sub(
          total_num_indirect_calls);
      break;
    }
    ++block_reactivation_requests_by_block[0].num_reactivations_scheduled;
    block_exec_acc[block_idx] =
        block_reactivation_requests_by_block[0].runtime_blocks_reactivated[i];
    const auto &lambda =
        *mem_pool_acc[0].deref_handle(block_reactivation_requests_by_block[0]
                                          .runtime_blocks_reactivated[i]
                                          .lambda);
    max_num_instructions_atomic.fetch_max(lambda.instructions.get_count());
  }
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    setup_block(
        PortableMemPool::DeviceAccessor_t mem_pool_acc,
        const typename Buffers::IndirectCallAccessorType indirect_call_acc,
        const cl::sycl::accessor<typename RuntimeBlockType::BlockMetadata, 1,
                                 cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer>
            block_exec_acc,
        const Index_t lambda_idx, const Index_t thread_idx) {
  const auto &indirect_call_reqs = indirect_call_acc[lambda_idx];
  if (thread_idx >= indirect_call_reqs.num_indirect_calls_allocated) {
    return;
  }
  const auto &indirect_call_req =
      indirect_call_reqs.indirect_call_requests[thread_idx];
  const auto block_idx =
      indirect_call_reqs
          .block_metadata_indexes[thread_idx /
                                  RuntimeBlockType::NumThreadsPerBlock];
  const auto thread_idx_in_block =
      thread_idx % RuntimeBlockType::NumThreadsPerBlock;
  auto &target_block =
      *mem_pool_acc[0].deref_handle(block_exec_acc[block_idx].block);
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
  append(mem_pool_acc[0].deref_handle(block)->block_metadata());
}

template <typename RuntimeBlockType, Index_t MaxNumIndirectCalls,
          Index_t MaxNumReactivations>
void IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::BlockReactivationRequestBuffer::
    append(const typename RuntimeBlockType::BlockMetadata &block_metadata) {
  cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      count(num_runtime_blocks_reactivated);
  const auto target_idx = count.fetch_add(1U);
  runtime_blocks_reactivated[target_idx] = block_metadata;
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
  block_reactivation_requests_by_block[0].num_reactivations_scheduled = 0;
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
auto IndirectCallHandler<RuntimeBlockType, MaxNumIndirectCalls,
                         MaxNumReactivations>::
    populate_block_exec_group(
        cl::sycl::queue &work_queue,
        cl::sycl::buffer<PortableMemPool> &mem_pool_buffer, Buffers &buffers,
        const Program program) -> BlockExecGroup {
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
    auto total_indirect_calls_across_scheduled_blocks_acc =
        buffers.total_indirect_calls_across_scheduled_blocks
            .template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<class ResetIndirectCallHandlerBuffers>([=] {
      max_num_threads_per_lambda_acc[0] = 0;
      max_num_instructions_acc[0] = 0;
      num_blocks_scheduled_acc[0] = 0;
      total_indirect_calls_across_scheduled_blocks_acc[0] = 0;
    });
  });

  // If there are any host-side buffered calls and space exists in indirect call
  // buffer, append queued calls.
  {
    std::optional<cl::sycl::accessor<IndirectCallRequestBuffer, 1,
                                     cl::sycl::access::mode::read_write,
                                     cl::sycl::access::target::host_buffer>>
        indirect_call_acc;
    for (Index_t lambda_idx = 0; lambda_idx < program.get_count();
         ++lambda_idx) {
      auto &buffered_indirect_call_requests =
          buffers.buffered_indirect_call_requests[lambda_idx];
      if (buffered_indirect_call_requests.empty()) {
        continue;
      }
      if (!indirect_call_acc.has_value()) {
        indirect_call_acc =
            buffers.indirect_call_requests_by_block
                .template get_access<cl::sycl::access::mode::read_write>();
      }
      auto &indirect_call_reqs = (*indirect_call_acc)[lambda_idx];
      while (!buffered_indirect_call_requests.empty() &&
             indirect_call_reqs.num_indirect_call_reqs < MaxNumIndirectCalls) {
        indirect_call_reqs.append(buffered_indirect_call_requests.front());
        buffered_indirect_call_requests.pop_front();
      }
    }
  }
  // If there are any host-side buffered reactivations and space exists in
  // reactivation buffer, append queued reactivations.
  if (!buffers.buffered_reactivations.empty()) {
    auto block_reactivation_requests_by_block_acc =
        buffers.block_reactivation_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>();
    while (!buffers.buffered_reactivations.empty() &&
           block_reactivation_requests_by_block_acc[0]
                   .num_runtime_blocks_reactivated < MaxNumReactivations) {
      block_reactivation_requests_by_block_acc[0].append(
          buffers.buffered_reactivations.front());
      buffers.buffered_reactivations.pop_front();
    }
  }

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
    auto total_indirect_calls_across_scheduled_blocks_acc =
        buffers.total_indirect_calls_across_scheduled_blocks
            .template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class BlockGroupsPerLambda>(
        cl::sycl::range<1>(program.get_count() + 1),
        [mem_pool_write, block_exec_group_acc, program,
         indirect_call_requests_by_block_acc,
         block_reactivation_requests_by_block_acc,
         atomic_max_threads_per_lambda, max_num_instructions_acc,
         num_blocks_scheduled_acc,
         total_indirect_calls_across_scheduled_blocks_acc](
            cl::sycl::item<1> itm) {
          const auto lambda_idx = itm.get_linear_id();
          if (lambda_idx < program.get_count()) {
            setup_block_for_lambda(
                mem_pool_write, indirect_call_requests_by_block_acc,
                block_exec_group_acc, atomic_max_threads_per_lambda,
                num_blocks_scheduled_acc, max_num_instructions_acc, lambda_idx,
                program, total_indirect_calls_across_scheduled_blocks_acc);
          } else {
            setup_block_for_reactivation(
                mem_pool_write, block_reactivation_requests_by_block_acc,
                block_exec_group_acc, num_blocks_scheduled_acc,
                max_num_instructions_acc,
                total_indirect_calls_across_scheduled_blocks_acc);
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
          cl::sycl::range<2>(program.get_count(),
                             max_num_threads_per_lambda_val),
          [mem_pool_write, block_exec_group_acc,
           indirect_call_requests_by_block_acc](const cl::sycl::item<2> itm) {
            setup_block(mem_pool_write, indirect_call_requests_by_block_acc,
                        block_exec_group_acc, itm.get_id(0), itm.get_id(1));
          });
    });
  }

  // Move unscheduled calls to a separate buffer. TODO optimize to avoid device
  // to host transfer in common case.
  {
    auto indirect_call_requests_by_block_acc =
        buffers.indirect_call_requests_by_block
            .template get_access<cl::sycl::access::mode::read>();
    for (Index_t lambda_idx = 0; lambda_idx < program.get_count();
         ++lambda_idx) {
      const auto &indirect_call_reqs =
          indirect_call_requests_by_block_acc[lambda_idx];
      for (Index_t thread_idx = indirect_call_reqs.num_indirect_calls_allocated;
           thread_idx < indirect_call_reqs.num_indirect_call_reqs;
           ++thread_idx) {
        const auto &req = indirect_call_reqs.indirect_call_requests[thread_idx];
        buffers.buffered_indirect_call_requests[lambda_idx].push_back(req);
      }
    }
  }

  {
    auto reactivation_buffer_acc =
        buffers.block_reactivation_requests_by_block
            .template get_access<cl::sycl::access::mode::read_write>();
    for (Index_t pending_reactivation_idx =
             reactivation_buffer_acc[0].num_reactivations_scheduled;
         pending_reactivation_idx <
         reactivation_buffer_acc[0].num_runtime_blocks_reactivated;
         ++pending_reactivation_idx) {
      buffers.buffered_reactivations.push_back(
          reactivation_buffer_acc[0]
              .runtime_blocks_reactivated[pending_reactivation_idx]);
    }
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
        [mem_pool_acc, program, indirect_call_requests_by_block_acc,
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
    return BlockExecGroup{.num_blocks = std::min(num_blocks_scheduled_acc[0],
                                                 MAX_BLOCKS_SCHEDULED_PER_PASS),
                          .max_num_instructions = max_num_instructions_acc[0]};
  }
}
} // namespace FunGPU::EvaluatorV2
