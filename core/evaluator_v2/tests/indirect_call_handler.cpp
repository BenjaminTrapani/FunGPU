#define BOOST_TEST_MODULE RuntimeBlockTestsModule

#include "core/evaluator_v2/indirect_call_handler.hpp"
#include "core/evaluator_v2/block_exec_group.hpp"
#include "core/evaluator_v2/compile_program.hpp"
#include "core/evaluator_v2/lambda.hpp"
#include "core/evaluator_v2/runtime_block.hpp"
#include "core/evaluator_v2/runtime_value.hpp"
#include "core/portable_mem_pool.hpp"
#include <boost/test/tools/context.hpp>
#include <boost/test/unit_test.hpp>
#include <unordered_set>

namespace FunGPU::EvaluatorV2 {
namespace {
using RuntimeBlockType = RuntimeBlock<64, 32>;

template <Index_t MAX_NUM_INDIRECT_CALLS, Index_t MAX_NUM_REACTIVATIONS>
struct Fixture {
  using IndirectCallHandlerType =
      IndirectCallHandler<RuntimeBlockType, MAX_NUM_INDIRECT_CALLS,
                          MAX_NUM_REACTIVATIONS>;

  Fixture(const std::string &program_path)
      : program(compile_program(
            program_path, RuntimeBlockType::NumThreadsPerBlock,
            RuntimeBlockType::NumThreadsPerBlock, mem_pool_buffer)) {
    std::cout
        << "Running on "
        << work_queue.get_device().get_info<cl::sycl::info::device::name>()
        << ", block size: " << sizeof(RuntimeBlockType)
        << ", max_blocks_allocated_per_pass="
        << IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS
        << ", max_blocks scheduled per pass="
        << IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS << std::endl;
  }

  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  const Program program;
  IndirectCallHandlerType::Buffers buffers{program.get_count()};
  cl::sycl::queue work_queue;
};

template <Index_t MAX_NUM_INDIRECT_CALLS = 512,
          Index_t MAX_NUM_REACTIVATIONS = 512>
struct BasicFixture
    : public Fixture<MAX_NUM_INDIRECT_CALLS, MAX_NUM_REACTIVATIONS> {
  BasicFixture()
      : Fixture<MAX_NUM_INDIRECT_CALLS, MAX_NUM_REACTIVATIONS>(
            "./test_programs/SimpleCall.fgpu") {}
};

BOOST_FIXTURE_TEST_CASE(basic, BasicFixture<>) {
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlockType>> caller_buf(
      cl::sycl::range<1>(1));
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlockType>>
      reactivated_block_buf(cl::sycl::range<1>(1));
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto caller_acc = caller_buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto reactivated_acc =
        reactivated_block_buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto block_reactivation_buffer_ac =
        buffers.block_reactivation_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    auto ind_call_reqs_by_block_acc =
        buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    const auto tmp_program = program;
    cgh.single_task<class RequestIndirectCall>([mem_pool_acc, tmp_program,
                                                caller_acc, reactivated_acc,
                                                block_reactivation_buffer_ac,
                                                ind_call_reqs_by_block_acc] {
      auto pre_allocated_rvs = RuntimeBlockType::pre_allocate_runtime_values<
          cl::sycl::access::target::device>(1, mem_pool_acc, tmp_program, 0);
      const auto mock_caller = mem_pool_acc[0].alloc<RuntimeBlockType>(
          tmp_program.element_handle(0), *pre_allocated_rvs, 1);
      caller_acc[0] = mock_caller;
      const auto captures_acc = mem_pool_acc[0].alloc_array<RuntimeValue>(2);
      auto *captures_data = mem_pool_acc[0].deref_handle(captures_acc);
      captures_data[0] = RuntimeValue(2.f);
      captures_data[1] = RuntimeValue(3.f);
      FunctionValue function_value(0, captures_acc);
      const auto test_args = mem_pool_acc[0].alloc_array<RuntimeValue>(1);
      RuntimeValue &arg_val = mem_pool_acc[0].deref_handle(test_args)[0];
      arg_val.data.float_val = 42;
      IndirectCallHandlerType::on_indirect_call(ind_call_reqs_by_block_acc,
                                                mock_caller, function_value, 1,
                                                2, test_args);

      auto pre_allocated_rvs_for_reactivated_block =
          RuntimeBlockType::pre_allocate_runtime_values<
              cl::sycl::access::target::device>(2, mem_pool_acc, tmp_program,
                                                1);
      const auto reactivated_block = mem_pool_acc[0].alloc<RuntimeBlockType>(
          tmp_program.element_handle(1),
          *pre_allocated_rvs_for_reactivated_block, 2);
      reactivated_acc[0] = reactivated_block;
      IndirectCallHandlerType::on_activate_block(
          mem_pool_acc, block_reactivation_buffer_ac, reactivated_block);
    });
  });

  const auto exec_group = IndirectCallHandlerType::populate_block_exec_group(
      work_queue, mem_pool_buffer, buffers, program);
  BOOST_CHECK_EQUAL(2, exec_group.num_blocks);
  BOOST_CHECK_EQUAL(3, exec_group.max_num_instructions);
  auto mem_pool_acc =
      mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
  auto block_descs =
      buffers.block_exec_group.get_access<cl::sycl::access::mode::read>();
  const auto validate_reactivation = [&](const auto block_idx) {
    const auto &block_meta_for_reactivation = block_descs[block_idx];
    BOOST_REQUIRE(block_meta_for_reactivation.lambda !=
                  PortableMemPool::Handle<Lambda>());
    BOOST_REQUIRE(block_meta_for_reactivation.block !=
                  PortableMemPool::Handle<RuntimeBlockType>());
    BOOST_CHECK(
        reactivated_block_buf.get_access<cl::sycl::access::mode::read>()[0] ==
        block_meta_for_reactivation.block);
    BOOST_CHECK(
        &mem_pool_acc[0].deref_handle(program)[1] ==
        mem_pool_acc[0].deref_handle(block_meta_for_reactivation.lambda));
    BOOST_CHECK_EQUAL(2, block_meta_for_reactivation.num_threads);
  };

  const auto validate_call = [&](const auto block_idx) {
    auto block_descs =
        buffers.block_exec_group.get_access<cl::sycl::access::mode::read>();
    const auto &block_meta = block_descs[block_idx];
    BOOST_REQUIRE(block_meta.lambda != PortableMemPool::Handle<Lambda>());
    BOOST_REQUIRE(block_meta.block !=
                  PortableMemPool::Handle<RuntimeBlockType>());
    const auto &first_block = *mem_pool_acc[0].deref_handle(block_meta.block);
    BOOST_CHECK_EQUAL(1, first_block.num_threads);
    const std::array<RuntimeValue, 3> expected_captures_and_arg{
        RuntimeValue(2), RuntimeValue(3), RuntimeValue(42)};
    for (Index_t i = 0; i < expected_captures_and_arg.size(); ++i) {
      const auto actual_val = first_block.registers[0][i];
      const auto expected_val = expected_captures_and_arg[i];
      BOOST_CHECK_EQUAL(expected_val.data.float_val, actual_val.data.float_val);
    }
    BOOST_CHECK(mem_pool_acc[0].deref_handle(block_meta.lambda) ==
                &mem_pool_acc[0].deref_handle(program)[0]);
    BOOST_CHECK_EQUAL(1, block_meta.num_threads);
    const auto &target_data = first_block.target_data[0];
    BOOST_CHECK_EQUAL(1, target_data.thread);
    BOOST_CHECK_EQUAL(2, target_data.register_idx);
    BOOST_CHECK(caller_buf.get_access<cl::sycl::access::mode::read>()[0] ==
                target_data.block);
  };

  if (block_descs[0].lambda == program.element_handle(0)) {
    validate_call(0);
    validate_reactivation(1);
  } else {
    validate_call(1);
    validate_reactivation(0);
  }
}

using BasicFixtureBlockCountIndirectCallsLimitted = BasicFixture<128, 256>;
BOOST_FIXTURE_TEST_CASE(extra_block_reactivations_buffered_until_next_pass,
                        BasicFixtureBlockCountIndirectCallsLimitted) {
  const auto tmp_program = program;
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_write =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto block_reactivation_buffer_ac =
        buffers.block_reactivation_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class ReactivateBlocks>(
        cl::sycl::range<1>(
            IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS + 1),
        [=](const cl::sycl::item<1> itm) {
          auto pre_allocated_rvs =
              RuntimeBlockType::pre_allocate_runtime_values<
                  cl::sycl::access::target::device>(1, mem_pool_write,
                                                    tmp_program, 0);
          const auto reactivated_block =
              mem_pool_write[0].alloc<RuntimeBlockType>(
                  tmp_program.element_handle(1), *pre_allocated_rvs,
                  itm.get_linear_id());
          IndirectCallHandlerType::on_activate_block(
              mem_pool_write, block_reactivation_buffer_ac, reactivated_block);
        });
  });

  const auto exec_group = IndirectCallHandlerType::populate_block_exec_group(
      work_queue, mem_pool_buffer, buffers, program);
  BOOST_CHECK_EQUAL(IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS,
                    exec_group.num_blocks);
  std::set<Index_t> observed_block_thread_counts;
  const auto add_all_thread_counts_for_exec_group =
      [&](const BlockExecGroup &block_exec_group) {
        auto block_exec_group_acc =
            buffers.block_exec_group.get_access<cl::sycl::access::mode::read>();
        for (Index_t i = 0; i < block_exec_group.num_blocks; ++i) {
          BOOST_CHECK(observed_block_thread_counts
                          .emplace(block_exec_group_acc[i].num_threads)
                          .second);
        }
      };
  add_all_thread_counts_for_exec_group(exec_group);
  const auto second_exec_group =
      IndirectCallHandlerType::populate_block_exec_group(
          work_queue, mem_pool_buffer, buffers, program);
  BOOST_CHECK_EQUAL(1, second_exec_group.num_blocks);
  add_all_thread_counts_for_exec_group(second_exec_group);
  BOOST_CHECK_EQUAL(IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS + 1,
                    observed_block_thread_counts.size());
}

using BasicFixtureBlockCountReactivationLimitted =
    BasicFixture<512, 512 / RuntimeBlockType::NumThreadsPerBlock - 2>;
BOOST_FIXTURE_TEST_CASE(extra_indirect_call_requests_buffered_until_next_pass,
                        BasicFixtureBlockCountReactivationLimitted) {
  const auto tmp_program = program;
  static constexpr auto TOTAL_NUM_INDIRECT_CALLS =
      IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS *
          RuntimeBlockType::NumThreadsPerBlock +
      1;
  static_assert(TOTAL_NUM_INDIRECT_CALLS <=
                IndirectCallHandlerType::MAX_NUM_INDIRECT_CALLS);
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto ind_call_reqs_by_block_acc =
        buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class QueueIndirectCalls>(
        cl::sycl::range<1>(TOTAL_NUM_INDIRECT_CALLS),
        [=](const cl::sycl::item<1> idx) {
          auto pre_allocated_rvs =
              RuntimeBlockType::pre_allocate_runtime_values<
                  cl::sycl::access::target::device>(1, mem_pool_acc,
                                                    tmp_program, 0);
          const auto mock_caller = mem_pool_acc[0].alloc<RuntimeBlockType>(
              tmp_program.element_handle(0), *pre_allocated_rvs, 1);
          FunctionValue function_value(
              1, PortableMemPool::ArrayHandle<RuntimeValue>());
          const auto test_args = mem_pool_acc[0].alloc_array<RuntimeValue>(1);
          IndirectCallHandlerType::on_indirect_call(
              ind_call_reqs_by_block_acc, mock_caller, function_value,
              idx.get_linear_id(), 2, test_args);
        });
  });

  std::set<Index_t> observed_block_thread_indices;
  const auto add_all_thread_indices_for_indirect_calls =
      [&](const BlockExecGroup &block_exec_group) {
        auto block_exec_group_acc =
            buffers.block_exec_group.get_access<cl::sycl::access::mode::read>();
        auto mem_pool_acc =
            mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
        for (Index_t i = 0; i < block_exec_group.num_blocks; ++i) {
          const auto &runtime_block =
              *mem_pool_acc[0].deref_handle(block_exec_group_acc[i].block);
          for (Index_t thread_idx = 0;
               thread_idx < block_exec_group_acc[i].num_threads; ++thread_idx) {
            BOOST_CHECK(
                observed_block_thread_indices
                    .emplace(runtime_block.target_data[thread_idx].thread)
                    .second);
          }
        }
      };
  const auto first_block_exec_group =
      IndirectCallHandlerType::populate_block_exec_group(
          work_queue, mem_pool_buffer, buffers, program);
  BOOST_CHECK_EQUAL(IndirectCallHandlerType::MAX_BLOCKS_SCHEDULED_PER_PASS,
                    first_block_exec_group.num_blocks);
  add_all_thread_indices_for_indirect_calls(first_block_exec_group);

  const auto second_block_exec_group =
      IndirectCallHandlerType::populate_block_exec_group(
          work_queue, mem_pool_buffer, buffers, program);
  BOOST_CHECK_EQUAL(1, second_block_exec_group.num_blocks);
  add_all_thread_indices_for_indirect_calls(second_block_exec_group);
  BOOST_CHECK_EQUAL(TOTAL_NUM_INDIRECT_CALLS,
                    observed_block_thread_indices.size());
}

struct AdvancedFixture : public Fixture<512, 512> {
  AdvancedFixture() : Fixture("./test_programs/ListExample.fgpu") {}
};

// This one exercises multiple blocks, grouping of indirect call requests by
// lambda and concurrent merging.
BOOST_FIXTURE_TEST_CASE(advanced, AdvancedFixture) {
  // test invocation of lambdas 7, 8 and 9. Incomplete block for 7, 2 blocks for
  // 8, 3 and 1 thread for 9
  PortableMemPool::Handle<RuntimeBlockType> caller;
  {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    auto pre_allocated_rvs = RuntimeBlockType::pre_allocate_runtime_values<
        cl::sycl::access::target::host_buffer>(1, mem_pool_acc, program, 0);
    BOOST_REQUIRE(pre_allocated_rvs.has_value());
    caller = mem_pool_acc[0].alloc<RuntimeBlockType>(program.element_handle(0),
                                                     *pre_allocated_rvs, 1);
  }
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    const auto tmp_program = program;
    auto ind_call_reqs_by_block_acc =
        buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class RequestIndirectCalls>(
        cl::sycl::range<2>(3, RuntimeBlockType::NumThreadsPerBlock * 4),
        [mem_pool_acc, tmp_program, caller,
         ind_call_reqs_by_block_acc](const cl::sycl::item<2> itm) {
          switch (itm.get_id(0)) {
          case 0:
            if (itm.get_id(1) >= RuntimeBlockType::NumThreadsPerBlock / 2) {
              return;
            }
            break;
          case 1:
            if (itm.get_id(1) >= RuntimeBlockType::NumThreadsPerBlock * 2) {
              return;
            }
            break;
          case 2:
            if (itm.get_id(1) >= RuntimeBlockType::NumThreadsPerBlock * 3 + 1) {
              return;
            }
            break;
          }
          const auto captures_acc =
              mem_pool_acc[0].alloc_array<RuntimeValue>(itm.get_id(0) + 1);
          auto *captures_data = mem_pool_acc[0].deref_handle(captures_acc);
          const auto test_args =
              mem_pool_acc[0].alloc_array<RuntimeValue>(itm.get_id(0) + 4);
          auto *arg_vals = mem_pool_acc[0].deref_handle(test_args);
          for (Index_t i = 0; i < captures_acc.get_count(); ++i) {
            captures_data[i] = RuntimeValue(itm.get_id(0) + i);
          }
          for (Index_t i = 0; i < test_args.get_count(); ++i) {
            arg_vals[i] =
                RuntimeValue(itm.get_id(0) + i + captures_acc.get_count());
          }
          FunctionValue function_value(itm.get_id(0) + 7, captures_acc);
          IndirectCallHandlerType::on_indirect_call(
              ind_call_reqs_by_block_acc, caller, function_value, itm.get_id(0),
              itm.get_id(1), test_args);
        });
  });

  const auto exec_group = IndirectCallHandlerType::populate_block_exec_group(
      work_queue, mem_pool_buffer, buffers, program);
  BOOST_CHECK_EQUAL(7, exec_group.num_blocks);
  BOOST_CHECK_EQUAL(7, exec_group.max_num_instructions);
  auto mem_pool_acc =
      mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();

  const auto generate_set = [](const auto num_threads) {
    std::unordered_set<Index_t> result;
    for (Index_t idx = 0; idx < num_threads; ++idx) {
      result.emplace(idx);
    }
    return result;
  };

  std::map<Index_t, Index_t> num_blocks_per_lambda_idx;
  std::map<Index_t, std::unordered_set<Index_t>>
      expected_target_regs_per_lambda;
  expected_target_regs_per_lambda[0] =
      generate_set(RuntimeBlockType::NumThreadsPerBlock / 2);
  expected_target_regs_per_lambda[1] =
      generate_set(RuntimeBlockType::NumThreadsPerBlock * 2);
  expected_target_regs_per_lambda[2] =
      generate_set(RuntimeBlockType::NumThreadsPerBlock * 3 + 1);
  auto block_descs =
      buffers.block_exec_group.get_access<cl::sycl::access::mode::read>();
  std::cout << "num blocks: " << exec_group.num_blocks << std::endl;
  for (Index_t i = 0; i < exec_group.num_blocks; ++i) {
    const auto &block_meta = block_descs[i];
    BOOST_TEST_INFO_SCOPE("Block index " << i);
    BOOST_REQUIRE(block_meta.lambda != PortableMemPool::Handle<Lambda>());
    BOOST_REQUIRE(block_meta.block !=
                  PortableMemPool::Handle<RuntimeBlockType>());
    const auto *lambdas = mem_pool_acc[0].deref_handle(program);
    auto lambda_idx = 0;
    const auto &block_metadata_lambda =
        *mem_pool_acc[0].deref_handle(block_meta.lambda);
    if (&block_metadata_lambda == &lambdas[7]) {
      lambda_idx = 0;
    } else if (&block_metadata_lambda == &lambdas[8]) {
      lambda_idx = 1;
    } else if (&block_metadata_lambda == &lambdas[9]) {
      lambda_idx = 2;
    } else {
      std::optional<Index_t> lambda_idx_opt;
      for (auto searched_lambda_idx = 0;
           searched_lambda_idx < program.get_count(); ++searched_lambda_idx) {
        if (&block_metadata_lambda == &lambdas[searched_lambda_idx]) {
          lambda_idx_opt = searched_lambda_idx;
          break;
        }
      }
      BOOST_FAIL("Instructions not for one of expected lambdas: " +
                     (lambda_idx_opt.has_value()
                          ? std::to_string(lambda_idx_opt.value())
                          : "nullopt") +
                     " for block " + std::to_string(i)
                 << ", block thread count: " << block_meta.num_threads);
    }
    num_blocks_per_lambda_idx[lambda_idx]++;
    BOOST_CHECK(&block_metadata_lambda ==
                &mem_pool_acc[0].deref_handle(program)[lambda_idx + 7]);
    const auto &first_block = *mem_pool_acc[0].deref_handle(block_meta.block);
    std::vector<RuntimeValue> expected_captures_and_args;
    for (Index_t j = 0; j < (lambda_idx + 4) + (lambda_idx + 1); j++) {
      expected_captures_and_args.emplace_back(j + lambda_idx);
    }
    const auto expected_num_threads = [&] {
      switch (lambda_idx) {
      case 0:
        return RuntimeBlockType::NumThreadsPerBlock / 2;
      case 1:
        return RuntimeBlockType::NumThreadsPerBlock;
      case 2:
        return num_blocks_per_lambda_idx[2] <= 3
                   ? RuntimeBlockType::NumThreadsPerBlock
                   : 1;
      }
      BOOST_FAIL("Unexpected");
      return Index_t(0);
    }();
    BOOST_CHECK_EQUAL(expected_num_threads, block_meta.num_threads);
    BOOST_CHECK_EQUAL(block_meta.num_threads, first_block.num_threads);
    auto &expected_thread_indices = expected_target_regs_per_lambda[lambda_idx];
    for (Index_t tid = 0; tid < expected_num_threads; ++tid) {
      for (Index_t j = 0; j < expected_captures_and_args.size(); ++j) {
        const auto actual_val = first_block.registers[tid][j];
        const auto expected_val = expected_captures_and_args[j];
        BOOST_CHECK_EQUAL(expected_val.data.float_val,
                          actual_val.data.float_val);
      }
      const auto &target_data = first_block.target_data[tid];
      BOOST_CHECK_EQUAL(lambda_idx, target_data.thread);
      BOOST_CHECK_EQUAL(
          1, expected_thread_indices.erase(target_data.register_idx));
      BOOST_CHECK(caller == target_data.block);
    }
  }
  BOOST_CHECK_EQUAL(1, num_blocks_per_lambda_idx[0]);
  BOOST_CHECK_EQUAL(2, num_blocks_per_lambda_idx[1]);
  BOOST_CHECK_EQUAL(4, num_blocks_per_lambda_idx[2]);
  for (const auto &[k, remaining_target_regs] :
       expected_target_regs_per_lambda) {
    std::stringstream remaining_values;
    for (const auto target_reg : remaining_target_regs) {
      remaining_values << target_reg << ", ";
    }
    BOOST_TEST_INFO("lambda "
                    << k << ", remaining values: " << remaining_values.str());
    BOOST_CHECK(remaining_target_regs.empty());
  }
}
} // namespace
} // namespace FunGPU::EvaluatorV2
