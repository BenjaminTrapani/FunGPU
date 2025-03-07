#define BOOST_TEST_MODULE BlockGeneratorTestsModule

#include "Core/EvaluatorV2/block_generator.hpp"
#include "Core/EvaluatorV2/compile_program.hpp"
#include "Core/block_prep.hpp"
#include "Core/visitor.hpp"
#include <boost/test/tools/context.hpp>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
namespace {
constexpr Index_t REGISTERS_PER_BLOCK = 64;

struct Fixture {
  void check_program_generates_instructions(
      const std::string &program_path,
      const std::vector<std::vector<Instruction>> &instructions) {
    const auto program =
        compile_program(program_path, REGISTERS_PER_BLOCK, 32, mem_pool_buffer);
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();

    BOOST_REQUIRE_EQUAL(instructions.size(), program.get_count());
    for (Index_t lambda_idx = 0; lambda_idx < program.get_count();
         ++lambda_idx) {
      const auto &lambda = mem_pool_acc[0].deref_handle(program)[lambda_idx];
      BOOST_REQUIRE_EQUAL(instructions[lambda_idx].size(),
                          lambda.instructions.get_count());
      for (Index_t instruction_idx = 0;
           instruction_idx < lambda.instructions.get_count();
           ++instruction_idx) {
        BOOST_TEST_INFO_SCOPE("lambda_idx: " << lambda_idx
                                             << ", instruction_idx: "
                                             << instruction_idx);
        BOOST_CHECK(instructions[lambda_idx][instruction_idx].equals(
            mem_pool_acc[0].deref_handle(lambda.instructions)[instruction_idx],
            mem_pool_acc));
      }
    }
  }

  PortableMemPool::ArrayHandle<Index_t>
  portable_index_array_from_vector(const std::vector<Index_t> &vec) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    auto array_handle = mem_pool_acc[0].alloc_array<Index_t>(vec.size());
    auto *captured_registers_data = mem_pool_acc[0].deref_handle(array_handle);
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
            BlockingCallIndirect{2, 0, portable_index_array_from_vector({1})})},
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
            BlockingCallIndirect{3, 1, portable_index_array_from_vector({2})})},
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
            BlockingCallIndirect{4, 1, portable_index_array_from_vector({3})})},
       {create_instruction(Add{2, 0, 1})}});
}

BOOST_FIXTURE_TEST_CASE(SimpleLetRec, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/SimpleLetRec.fgpu",
      {{create_instruction(CreateLambda{0, 1, {}}),
        create_instruction(AssignConstant{1, 0}),
        create_instruction(AssignConstant{2, 5}),
        create_instruction(BlockingCallIndirect{
            3, 0, portable_index_array_from_vector({0, 1, 2})})},
       {create_instruction(Equal{3, 1, 2}),
        create_instruction(
            CreateLambda{4, 2, portable_index_array_from_vector({0, 1, 2})}),
        create_instruction(If{3, 3, 4}), create_instruction(Assign{5, 2}),
        create_instruction(BlockingCallIndirect{
            6, 4, PortableMemPool::ArrayHandle<Index_t>()})},
       {create_instruction(AssignConstant{3, 1}),
        create_instruction(Add{4, 3, 1}),
        create_instruction(BlockingCallIndirect{
            5, 0, portable_index_array_from_vector({0, 4, 2})})}});
}

BOOST_FIXTURE_TEST_CASE(CheckBarrierInstructionGenerated, Fixture) {
  check_program_generates_instructions(
      "./TestPrograms/CallResultBoundInLet.fgpu",
      {{create_instruction(
            CreateLambda{0, 1, PortableMemPool::ArrayHandle<Index_t>()}),
        create_instruction(AssignConstant{1, 1}),
        create_instruction(AssignConstant{2, 2}),
        create_instruction(
            CallIndirect{3, 0, portable_index_array_from_vector({1})}),
        create_instruction(
            CallIndirect{4, 0, portable_index_array_from_vector({2})}),
        create_instruction(InstructionBarrier()),
        create_instruction(Add{5, 3, 4})},
       {create_instruction(AssignConstant{1, 1}),
        create_instruction(Add{2, 0, 1})}});
}

/*
BOOST_FIXTURE_TEST_CASE(MultiLetRec, Fixture) {
  check_program_generates_instructions("./TestPrograms/MultiLetRec.fgpu", {});
}*/
} // namespace
} // namespace FunGPU::EvaluatorV2
