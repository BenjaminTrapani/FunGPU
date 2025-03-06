#define BOOST_TEST_MODULE RuntimeBlockTestsModule
#include "Core/EvaluatorV2/indirect_call_handler.hpp"
#include "Core/EvaluatorV2/compile_program.hpp"
#include "Core/EvaluatorV2/runtime_block.hpp"
#include "Core/EvaluatorV2/runtime_value.hpp"
#include "Core/portable_mem_pool.hpp"
#include <boost/test/unit_test.hpp>
#include <unordered_set>

namespace FunGPU::EvaluatorV2 {
namespace {
struct Fixture {
  using RuntimeBlockType = RuntimeBlock<64, 32>;
  using IndirectCallHandlerType =
      IndirectCallHandler<RuntimeBlockType, 128, 128>;

  Fixture(const std::string &program_path)
      : program(compile_program(
            program_path, RuntimeBlockType::NumThreadsPerBlock,
            RuntimeBlockType::NumThreadsPerBlock, mem_pool_buffer)) {
    std::cout
        << "Running on "
        << work_queue.get_device().get_info<cl::sycl::info::device::name>()
        << ", block size: " << sizeof(RuntimeBlockType) << std::endl;
    buffers.update_for_num_lambdas(program.get_count());
  }

  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  const Program program;
  IndirectCallHandlerType::Buffers buffers{program.get_count()};
  std::shared_ptr<IndirectCallHandlerType> indirect_call_handler =
      std::make_shared<IndirectCallHandlerType>();
  cl::sycl::buffer<IndirectCallHandlerType> indirect_call_handler_buff{
      indirect_call_handler, cl::sycl::range<1>(1)};
  cl::sycl::queue work_queue;
};

struct BasicFixture : public Fixture {
  BasicFixture() : Fixture("./TestPrograms/SimpleCall.fgpu") {}
};

BOOST_FIXTURE_TEST_CASE(basic, BasicFixture) {
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlockType>> caller_buf(
      cl::sycl::range<1>(1));
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlockType>>
      reactivated_block_buf(cl::sycl::range<1>(1));
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto indirect_call_acc =
        indirect_call_handler_buff
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto caller_acc =
        caller_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto reactivated_acc =
        reactivated_block_buf.get_access<cl::sycl::access::mode::discard_write>(
            cgh);
    auto ind_call_reqs_by_block_acc =
        buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    const auto tmp_program = program;
    cgh.single_task<class RequestIndirectCall>([indirect_call_acc, mem_pool_acc,
                                                tmp_program, caller_acc,
                                                reactivated_acc,
                                                ind_call_reqs_by_block_acc] {
      const auto lambda_0 = mem_pool_acc[0].deref_handle(tmp_program)[0];
      auto pre_allocated_rvs = RuntimeBlockType::pre_allocate_runtime_values<
          cl::sycl::access::target::device>(1, mem_pool_acc, tmp_program, 0);
      const auto mock_caller = mem_pool_acc[0].alloc<RuntimeBlockType>(
          lambda_0.instructions, pre_allocated_rvs, 1);
      caller_acc[0] = mock_caller;
      const auto captures_acc = mem_pool_acc[0].alloc_array<RuntimeValue>(2);
      auto *captures_data = mem_pool_acc[0].deref_handle(captures_acc);
      captures_data[0] = RuntimeValue(2.f);
      captures_data[1] = RuntimeValue(3.f);
      FunctionValue function_value(0, captures_acc);
      const auto test_args = mem_pool_acc[0].alloc_array<RuntimeValue>(1);
      RuntimeValue &arg_val = mem_pool_acc[0].deref_handle(test_args)[0];
      arg_val.data.float_val = 42;
      indirect_call_acc[0].on_indirect_call(ind_call_reqs_by_block_acc,
                                            mock_caller, function_value, 1, 2,
                                            test_args);

      const auto lambda_1 = mem_pool_acc[0].deref_handle(tmp_program)[1];
      auto pre_allocated_rvs_for_reactivated_block =
          RuntimeBlockType::pre_allocate_runtime_values<
              cl::sycl::access::target::device>(2, mem_pool_acc, tmp_program,
                                                1);
      const auto reactivated_block = mem_pool_acc[0].alloc<RuntimeBlockType>(
          lambda_1.instructions, pre_allocated_rvs_for_reactivated_block, 2);
      reactivated_acc[0] = reactivated_block;
      indirect_call_acc[0].on_activate_block(mem_pool_acc, reactivated_block);
    });
  });

  const auto exec_group = IndirectCallHandlerType::create_block_exec_group(
      work_queue, mem_pool_buffer, indirect_call_handler_buff, buffers,
      program);
  BOOST_CHECK_EQUAL(2, exec_group.block_descs.get_count());
  BOOST_CHECK_EQUAL(3, exec_group.max_num_instructions);
  auto mem_pool_acc =
      mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();

  const auto validate_reactivation = [&](const auto block_idx) {
    const auto &block_meta_for_reactivation =
        mem_pool_acc[0].deref_handle(exec_group.block_descs)[block_idx];
    BOOST_REQUIRE(block_meta_for_reactivation.instructions !=
                  PortableMemPool::ArrayHandle<Instruction>());
    BOOST_REQUIRE(block_meta_for_reactivation.block !=
                  PortableMemPool::Handle<RuntimeBlockType>());
    BOOST_CHECK(
        reactivated_block_buf.get_access<cl::sycl::access::mode::read>()[0] ==
        block_meta_for_reactivation.block);
    BOOST_CHECK(mem_pool_acc[0].deref_handle(program)[1].instructions ==
                block_meta_for_reactivation.instructions);
    BOOST_CHECK_EQUAL(2, block_meta_for_reactivation.num_threads);
  };

  const auto validate_call = [&](const auto block_idx) {
    const auto &block_meta =
        mem_pool_acc[0].deref_handle(exec_group.block_descs)[block_idx];
    BOOST_REQUIRE(block_meta.instructions !=
                  PortableMemPool::ArrayHandle<Instruction>());
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
    BOOST_CHECK(block_meta.instructions ==
                mem_pool_acc[0].deref_handle(program)[0].instructions);
    BOOST_CHECK_EQUAL(1, block_meta.num_threads);
    const auto &target_data = first_block.target_data[0];
    BOOST_CHECK_EQUAL(1, target_data.thread);
    BOOST_CHECK_EQUAL(2, target_data.register_idx);
    BOOST_CHECK(caller_buf.get_access<cl::sycl::access::mode::read>()[0] ==
                target_data.block);
  };

  if (mem_pool_acc[0].deref_handle(exec_group.block_descs)[0].instructions ==
      mem_pool_acc[0].deref_handle(program)[0].instructions) {
    validate_call(0);
    validate_reactivation(1);
  } else {
    validate_call(1);
    validate_reactivation(0);
  }
}

struct AdvancedFixture : public Fixture {
  AdvancedFixture() : Fixture("./TestPrograms/ListExample.fgpu") {}
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
    caller = mem_pool_acc[0].alloc<RuntimeBlockType>(
        mem_pool_acc[0].deref_handle(program)[0].instructions,
        pre_allocated_rvs, 1);
  }
  work_queue.submit([&](cl::sycl::handler &cgh) {
    auto indirect_call_acc =
        indirect_call_handler_buff
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
    const auto tmp_program = program;
    auto ind_call_reqs_by_block_acc =
        buffers.indirect_call_requests_by_block
            .get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class RequestIndirectCalls>(
        cl::sycl::range<2>(3, RuntimeBlockType::NumThreadsPerBlock * 4),
        [indirect_call_acc, mem_pool_acc, tmp_program, caller,
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
          indirect_call_acc[0].on_indirect_call(
              ind_call_reqs_by_block_acc, caller, function_value, itm.get_id(0),
              itm.get_id(1), test_args);
        });
  });

  const auto exec_group = IndirectCallHandlerType::create_block_exec_group(
      work_queue, mem_pool_buffer, indirect_call_handler_buff, buffers,
      program);
  BOOST_CHECK_EQUAL(7, exec_group.block_descs.get_count());
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
  for (Index_t i = 0; i < exec_group.block_descs.get_count(); ++i) {
    const auto &block_meta =
        mem_pool_acc[0].deref_handle(exec_group.block_descs)[i];
    BOOST_REQUIRE(block_meta.instructions !=
                  PortableMemPool::ArrayHandle<Instruction>());
    BOOST_REQUIRE(block_meta.block !=
                  PortableMemPool::Handle<RuntimeBlockType>());
    const auto *lambdas = mem_pool_acc[0].deref_handle(program);
    auto lambda_idx = 0;
    if (block_meta.instructions == lambdas[7].instructions) {
      lambda_idx = 0;
    } else if (block_meta.instructions == lambdas[8].instructions) {
      lambda_idx = 1;
    } else if (block_meta.instructions == lambdas[9].instructions) {
      lambda_idx = 2;
    } else {
      BOOST_FAIL("Instructions not for one of expected lambdas");
    }
    num_blocks_per_lambda_idx[lambda_idx]++;
    BOOST_CHECK(
        block_meta.instructions ==
        mem_pool_acc[0].deref_handle(program)[lambda_idx + 7].instructions);
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
