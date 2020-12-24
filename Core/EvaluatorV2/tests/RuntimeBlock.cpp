#include <boost/test/tools/old/interface.hpp>
#include <cmath>
#define BOOST_TEST_MODULE RuntimeBlockTestsModule
#include <boost/test/included/unit_test.hpp>

#include "Core/BlockPrep.hpp"
#include "Core/Compiler.hpp"
#include "Core/EvaluatorV2/BlockGenerator.h"
#include "Core/EvaluatorV2/RuntimeBlock.hpp"
#include "Core/Parser.hpp"
#include "Core/Visitor.hpp"
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
        << std::endl;
  }

  Program generate_program(const std::string &program_path) {
    Parser parser(program_path);
    auto parsed_result = parser.ParseProgram();
    std::cout << "Parsed program: " << std::endl;
    parsed_result->DebugPrint(0);
    std::cout << std::endl;

    Compiler compiler(parsed_result, mem_pool_buffer);
    auto compiled_result = compiler.Compile();
    std::cout << "Compiled program: " << std::endl;
    compiler.DebugPrintAST(compiled_result);
    std::cout << std::endl;

    compiled_result = block_prep.PrepareForBlockGeneration(compiled_result);
    std::cout << "Program after prep for block generation: " << std::endl;
    compiler.DebugPrintAST(compiled_result);
    std::cout << std::endl << std::endl;

    const auto program = block_generator.construct_blocks(compiled_result);
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    std::cout << "Printed program: " << std::endl;
    std::cout << print(program, mem_pool_acc);
    return program;
  }

  void check_block_evaluates_to_value(
      const float_t val, const RuntimeBlockType::BlockMetadata block_meta) {
    work_queue.submit([&](cl::sycl::handler &cgh) {
      auto mem_pool_write =
          mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
      cl::sycl::accessor<RuntimeBlockType, 1,
                         cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          local_block(cl::sycl::range<1>(1), cgh);
      RuntimeBlockType::InstructionLocalMemAccessor local_instructions(
          cl::sycl::range<1>(block_meta.instructions.GetCount()), cgh);
      auto results_acc =
          results.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for<class TestEvalLoop>(
          cl::sycl::nd_range<1>(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
          [mem_pool_write, block_meta, local_block, local_instructions,
           results_acc](cl::sycl::nd_item<1> itm) {
            const auto thread_idx = itm.get_global_linear_id();
            if (thread_idx == 0) {
              local_block[0] = *mem_pool_write[0].derefHandle(block_meta.block);
            }
            const auto *instructions_global_data =
                mem_pool_write[0].derefHandle(block_meta.instructions);
            for (auto idx = thread_idx;
                 idx < block_meta.instructions.GetCount();
                 idx += THREADS_PER_BLOCK) {
              local_instructions[idx] = instructions_global_data[idx];
            }
            itm.barrier();
            RuntimeBlockType::Status status = local_block[0].evaluate(
                thread_idx, mem_pool_write, local_instructions,
                [](auto &&...) {}, [](const auto) {});
            if (status == RuntimeBlockType::Status::COMPLETE) {
              results_acc[thread_idx] =
                  local_block[0].result(thread_idx, local_instructions);
            }
          });
    });

    {
      const auto result_acc =
          results.get_access<cl::sycl::access::mode::read>();
      for (Index_t i = 0; i < THREADS_PER_BLOCK; ++i) {
        BOOST_REQUIRE(RuntimeValue::Type::FLOAT == result_acc[i].type);
        BOOST_CHECK_EQUAL(val, result_acc[i].data.float_val);
      }
    }
  }

  void check_first_lambda_evaluates_to(const float_t expected_val,
                                       const std::string &program) {
    const auto no_bindings_program = generate_program(program);
    BOOST_REQUIRE_EQUAL(1, no_bindings_program.GetCount());
    const auto block_meta = [&] {
      auto mem_pool_acc =
          mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
      const auto *lambdas = mem_pool_acc[0].derefHandle(no_bindings_program);
      const auto block_handle = mem_pool_acc[0].Alloc<RuntimeBlockType>();
      return RuntimeBlockType::BlockMetadata(block_handle,
                                             lambdas[0].instructions);
    }();
    check_block_evaluates_to_value(expected_val, block_meta);
  }

  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  BlockPrep block_prep{REGISTERS_PER_THREAD, 32, 32, mem_pool_buffer};
  BlockGenerator block_generator{mem_pool_buffer, REGISTERS_PER_THREAD};
  cl::sycl::queue work_queue;
  cl::sycl::buffer<RuntimeValue> results{cl::sycl::range<1>(THREADS_PER_BLOCK)};
};

BOOST_FIXTURE_TEST_CASE(EvaluateNoBindings, Fixture) {
  check_first_lambda_evaluates_to(14, "./TestPrograms/NoBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateMultiLet, Fixture) {
  check_first_lambda_evaluates_to(1, "./TestPrograms/MultiLet.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateComplexBindings, Fixture) {
  check_first_lambda_evaluates_to(13, "./TestPrograms/ComplexBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateFalseBranchInTailPos, Fixture) {
  check_first_lambda_evaluates_to(5,
                                  "./TestPrograms/FalseBranchInTailPos.fgpu");
}

BOOST_FIXTURE_TEST_CASE(EvaluateTrueBranchInTailPos, Fixture) {
  check_first_lambda_evaluates_to(4, "./TestPrograms/TrueBranchInTailPos.fgpu");
}
} // namespace
} // namespace FunGPU::EvaluatorV2
