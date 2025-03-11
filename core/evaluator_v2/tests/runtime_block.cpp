#define BOOST_TEST_MODULE RuntimeBlockTestsModule
#include "core/evaluator_v2/runtime_block.hpp"
#include "core/ast_node.hpp"
#include "core/block_prep.hpp"
#include "core/evaluator_v2/block_generator.hpp"
#include "core/evaluator_v2/compile_program.hpp"
#include "core/parser.hpp"
#include "core/visitor.hpp"
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
namespace {
constexpr Index_t REGISTERS_PER_THREAD = 64;
constexpr Index_t THREADS_PER_BLOCK = 32;

struct Fixture {
  using RuntimeBlockType =
      RuntimeBlock<REGISTERS_PER_THREAD, THREADS_PER_BLOCK>;

  Fixture() {
    std::cout
        << "Running on "
        << work_queue.get_device().get_info<cl::sycl::info::device::name>()
        << ", block size: " << sizeof(RuntimeBlockType) << std::endl;
  }

  Program generate_program(const std::string &program_path) {
    return compile_program(program_path, REGISTERS_PER_THREAD,
                           THREADS_PER_BLOCK, mem_pool_buffer);
  }

  void check_block_evaluates_to_value(
      const std::vector<float_t> vals,
      const RuntimeBlockType::BlockExecGroup block_group) {
    cl::sycl::buffer<RuntimeValue, 2> results(cl::sycl::range<2>(
        block_group.block_descs.get_count(), THREADS_PER_BLOCK));
    work_queue.submit([&](cl::sycl::handler &cgh) {
      auto mem_pool_write =
          mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
      cl::sycl::accessor<RuntimeBlockType, 1,
                         cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          local_block(cl::sycl::range<1>(1), cgh);
      RuntimeBlockType::InstructionLocalMemAccessor local_instructions(
          cl::sycl::range<2>(block_group.block_descs.get_count(),
                             block_group.max_num_instructions),
          cgh);
      auto results_acc =
          results.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for<class TestEvalLoop>(
          cl::sycl::nd_range<1>(THREADS_PER_BLOCK *
                                    block_group.block_descs.get_count(),
                                THREADS_PER_BLOCK),
          [mem_pool_write, block_group, local_block, local_instructions,
           results_acc](cl::sycl::nd_item<1> itm) {
            const auto thread_idx = itm.get_local_linear_id();
            const auto block_idx = itm.get_group_linear_id();
            const auto block_meta = mem_pool_write[0].deref_handle(
                block_group.block_descs)[block_idx];
            if (thread_idx == 0) {
              local_block[0] =
                  *mem_pool_write[0].deref_handle(block_meta.block);
            }
            const auto *instructions_global_data =
                mem_pool_write[0].deref_handle(block_meta.instructions);
            auto instructions_for_block = local_instructions[block_idx];
            for (auto idx = thread_idx;
                 idx < block_meta.instructions.get_count();
                 idx += THREADS_PER_BLOCK) {
              instructions_for_block[idx] = instructions_global_data[idx];
            }
            itm.barrier(cl::sycl::access::fence_space::local_space);
            RuntimeBlockType::Status status = local_block[0].evaluate(
                block_idx, thread_idx, itm, mem_pool_write, local_instructions,
                block_meta.instructions.get_count(), block_meta.num_threads,
                [](auto &&...) {}, [](const auto) {});
            if (status == RuntimeBlockType::Status::COMPLETE) {
              results_acc[cl::sycl::id<2>(block_idx, thread_idx)] =
                  local_block[0].result(block_idx, thread_idx,
                                        local_instructions,
                                        block_meta.instructions.get_count());
            }
          });
    });

    {
      const auto result_acc =
          results.get_access<cl::sycl::access::mode::read>();
      for (Index_t block_idx = 0;
           block_idx < block_group.block_descs.get_count(); ++block_idx) {
        for (Index_t i = 0; i < THREADS_PER_BLOCK; ++i) {
          const auto result_idx = cl::sycl::id<2>(block_idx, i);
          BOOST_CHECK_EQUAL(vals[block_idx],
                            result_acc[result_idx].data.float_val);
        }
      }
    }
  }

  void check_first_lambda_evaluates_to(const float_t expected_val,
                                       const std::string &program) {
    const auto no_bindings_program = generate_program(program);
    BOOST_REQUIRE_EQUAL(1, no_bindings_program.get_count());
    const auto block_exec_group = [&] {
      auto mem_pool_acc =
          mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
      const auto *lambdas = mem_pool_acc[0].deref_handle(no_bindings_program);
      const auto pre_allocated_rvs =
          RuntimeBlockType::pre_allocate_runtime_values(
              THREADS_PER_BLOCK, mem_pool_acc, no_bindings_program, 0);
      const auto block_handle = mem_pool_acc[0].alloc<RuntimeBlockType>(
          lambdas[0].instructions, pre_allocated_rvs, THREADS_PER_BLOCK);
      const auto block_metadata_array =
          mem_pool_acc[0].alloc_array<RuntimeBlockType::BlockMetadata>(1);
      auto *block_meta_array_data =
          mem_pool_acc[0].deref_handle(block_metadata_array);
      block_meta_array_data[0] = RuntimeBlockType::BlockMetadata(
          block_handle, lambdas[0].instructions, THREADS_PER_BLOCK);
      return RuntimeBlockType::BlockExecGroup(
          block_metadata_array, lambdas[0].instructions.get_count());
    }();

    check_block_evaluates_to_value({expected_val}, block_exec_group);
  }

  void check_coscheduled_blocks_work(
      const std::vector<std::pair<float_t, std::string>>
          &expected_value_per_program) {
    const auto block_exec_group = [&] {
      auto mem_pool_acc =
          mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
      const auto block_metadata_array =
          mem_pool_acc[0].alloc_array<RuntimeBlockType::BlockMetadata>(
              expected_value_per_program.size());
      Index_t max_instructions_per_block = 0;
      Index_t i = 0;
      auto *block_metadata_array_data =
          mem_pool_acc[0].deref_handle(block_metadata_array);
      for (const auto &[val, prog] : expected_value_per_program) {
        const auto no_bindings_program = generate_program(prog);
        BOOST_REQUIRE_EQUAL(1, no_bindings_program.get_count());
        const auto *lambdas = mem_pool_acc[0].deref_handle(no_bindings_program);
        const auto pre_allocate_runtime_values =
            RuntimeBlockType::pre_allocate_runtime_values(
                THREADS_PER_BLOCK, mem_pool_acc, no_bindings_program, 0);
        const auto block_handle = mem_pool_acc[0].alloc<RuntimeBlockType>(
            lambdas[0].instructions, pre_allocate_runtime_values,
            THREADS_PER_BLOCK);
        BOOST_REQUIRE(block_handle !=
                      PortableMemPool::Handle<RuntimeBlockType>());
        block_metadata_array_data[i++] = RuntimeBlockType::BlockMetadata(
            block_handle, lambdas[0].instructions, THREADS_PER_BLOCK);
        max_instructions_per_block = std::max(
            max_instructions_per_block, lambdas[0].instructions.get_count());
      }
      return RuntimeBlockType::BlockExecGroup(block_metadata_array,
                                              max_instructions_per_block);
    }();
    std::vector<float_t> expected_values;
    for (const auto &[val, prog] : expected_value_per_program) {
      expected_values.emplace_back(val);
    }
    check_block_evaluates_to_value(expected_values, block_exec_group);
  }

  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  BlockPrep block_prep{REGISTERS_PER_THREAD, 32, 32, mem_pool_buffer};
  BlockGenerator block_generator{mem_pool_buffer, REGISTERS_PER_THREAD};
  cl::sycl::queue work_queue;
};

BOOST_FIXTURE_TEST_CASE(EvaluateNoBindings, Fixture) {
  check_first_lambda_evaluates_to(14, "./test_programs/NoBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateMultiLet, Fixture) {
  check_first_lambda_evaluates_to(1, "./test_programs/MultiLet.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateComplexBindings, Fixture) {
  check_first_lambda_evaluates_to(13, "./test_programs/ComplexBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateFalseBranchInTailPos, Fixture) {
  check_first_lambda_evaluates_to(5,
                                  "./test_programs/FalseBranchInTailPos.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateTrueBranchInTailPos, Fixture) {
  check_first_lambda_evaluates_to(4,
                                  "./test_programs/TrueBranchInTailPos.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateBranchBlocksInParallelBasic, Fixture) {
  check_coscheduled_blocks_work(
      {{5, "./test_programs/FalseBranchInTailPos.fgpu"},
       {4, "./test_programs/TrueBranchInTailPos.fgpu"}});
}

BOOST_FIXTURE_TEST_CASE(EvaluateBranchBlocksInParallelAdvanced, Fixture) {
  check_coscheduled_blocks_work(
      {{5, "./test_programs/FalseBranchInTailPos.fgpu"},
       {4, "./test_programs/TrueBranchInTailPos.fgpu"},
       {13, "./test_programs/ComplexBindings.fgpu"}});
}
} // namespace
} // namespace FunGPU::EvaluatorV2
