#define BOOST_TEST_MODULE BlockGeneratorTestsModule
#include <boost/test/included/unit_test.hpp>

#include "Core/BlockPrep.hpp"
#include "Core/Compiler.hpp"
#include "Core/EvaluatorV2/BlockGenerator.h"
#include "Core/Parser.hpp"
#include "Core/Visitor.hpp"
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
namespace {
constexpr Index_t REGISTERS_PER_BLOCK = 64;

struct Fixture {
  void check_program_generates_instructions(
      const std::string &program_path,
      const std::vector<std::vector<Instruction>> &instructions) {
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

    BOOST_REQUIRE_EQUAL(instructions.size(), program.GetCount());
    for (Index_t lambda_idx = 0; lambda_idx < program.GetCount();
         ++lambda_idx) {
      const auto &lambda = mem_pool_acc[0].derefHandle(program)[lambda_idx];
      BOOST_REQUIRE_EQUAL(instructions[lambda_idx].size(),
                          lambda.instructions.GetCount());
      for (Index_t instruction_idx = 0;
           instruction_idx < lambda.instructions.GetCount();
           ++instruction_idx) {
        BOOST_CHECK(instructions[lambda_idx][instruction_idx].equals(
            mem_pool_acc[0].derefHandle(lambda.instructions)[instruction_idx],
            mem_pool_acc));
      }
    }
  }

  PortableMemPool::ArrayHandle<Index_t>
  portable_index_array_from_vector(const std::vector<Index_t> &vec) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    auto array_handle = mem_pool_acc[0].AllocArray<Index_t>(vec.size());
    auto *captured_registers_data = mem_pool_acc[0].derefHandle(array_handle);
    std::copy(vec.begin(), vec.end(), captured_registers_data);
    return array_handle;
  }

  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  BlockPrep block_prep{REGISTERS_PER_BLOCK, 32, 32, mem_pool_buffer};
  BlockGenerator block_generator{mem_pool_buffer, REGISTERS_PER_BLOCK};
};

template <typename InstructionType>
Instruction create_instruction(const InstructionType &value) {
  Instruction result;
  result.type = InstructionType::TYPE;
  visit(result,
        Visitor{[&](InstructionType &to_update) { to_update = value; },
                [&](auto &) {
                  throw std::invalid_argument(
                      "Unexpected dispatched instruction type");
                }},
        [](auto &) {
          throw std::invalid_argument("Unexpected instruction type");
        });
  return result;
}

BOOST_FIXTURE_TEST_CASE(NoBindingsBasicTest, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/NoBindings.fgpu",
      {{create_instruction(AssignConstant{0, 4}),
        create_instruction(AssignConstant{1, 2}),
        create_instruction(AssignConstant{2, 3}),
        create_instruction(AssignConstant{3, 4}),
        create_instruction(Div{4, 0, 1}), create_instruction(Mul{5, 2, 3}),
        create_instruction(Add{6, 4, 5})}});
}

BOOST_FIXTURE_TEST_CASE(MultiLetTest, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/MultiLet.fgpu",
      {{create_instruction(AssignConstant{0, 1}),
        create_instruction(AssignConstant{1, 2}),
        create_instruction(AssignConstant{2, 3}),
        create_instruction(Add{3, 0, 1}), create_instruction(Div{4, 2, 3})}});
}

BOOST_FIXTURE_TEST_CASE(SimpleCallTest, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/SimpleCall.fgpu",
      {{create_instruction(
            CreateLambda{0, 1, PortableMemPool::ArrayHandle<Index_t>()}),
        create_instruction(AssignConstant{1, 42}),
        create_instruction(
            CallIndirect{2, 0, portable_index_array_from_vector({1})})},
       {create_instruction(Assign{1, 0})}});
}

BOOST_FIXTURE_TEST_CASE(SimpleClosureTest, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/SimpleLambda.fgpu",
      {{create_instruction(AssignConstant{0, 2}),
        create_instruction(
            CreateLambda{1, 1, portable_index_array_from_vector({0})}),
        create_instruction(AssignConstant{2, 3}),
        create_instruction(
            CallIndirect{3, 1, portable_index_array_from_vector({2})})},
       {create_instruction(Add{2, 0, 1})}});
}

BOOST_FIXTURE_TEST_CASE(CallMultiBinding, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/CallMultiBindings.fgpu",
      {{create_instruction(AssignConstant{0, 4}),
        create_instruction(
            CreateLambda{1, 1, portable_index_array_from_vector({0})}),
        create_instruction(AssignConstant{2, 10}),
        create_instruction(AssignConstant{3, 9}),
        create_instruction(
            CallIndirect{4, 1, portable_index_array_from_vector({3})})},
       {create_instruction(Add{2, 0, 1})}});
}

BOOST_FIXTURE_TEST_CASE(SimpleLetRec, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/SimpleLetRec.fgpu",
      {{create_instruction(
            CreateLambda{0, 1, portable_index_array_from_vector({0})}),
        create_instruction(AssignConstant{1, 0}),
        create_instruction(AssignConstant{2, 5}),
        create_instruction(
            CallIndirect{3, 0, portable_index_array_from_vector({1, 2})})},
       {create_instruction(Equal{3, 2, 1}),
        create_instruction(
            CreateLambda{4, 2, portable_index_array_from_vector({0, 2, 1})}),
        create_instruction(If{3, 3, 4}), create_instruction(Assign{5, 1}),
        create_instruction(
            CallIndirect{6, 4, PortableMemPool::ArrayHandle<Index_t>()})},
       {create_instruction(AssignConstant{3, 1}),
        create_instruction(Add{4, 3, 1}),
        create_instruction(
            CallIndirect{5, 0, portable_index_array_from_vector({4, 2})})}});
}

/*
BOOST_FIXTURE_TEST_CASE(MultiLetRec, Fixture) {
  check_program_generates_instructions("./TestPrograms/MultiLetRec.fgpu", {});
}*/
} // namespace
} // namespace FunGPU::EvaluatorV2
