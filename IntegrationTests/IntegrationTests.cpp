//
// Created by benjamintrapani on 7/31/19.
//

#define BOOST_TEST_MODULE IntegrationTestsModule
#include <boost/test/unit_test.hpp>

// windows weirdness
#undef max
#undef min

#include "Core/CPUEvaluator.hpp"
#include "Core/Compiler.hpp"
#include "Core/Parser.hpp"
#include "Core/PortableMemPool.hpp"
#include "Core/BlockPrep.hpp"

using namespace FunGPU;

struct Fixture {
  Fixture() {
    if (mem_pool_data == nullptr) {
      mem_pool_data = std::make_shared<PortableMemPool>();
      mem_pool_buffer = std::make_shared<cl::sycl::buffer<PortableMemPool>>(
          mem_pool_data, cl::sycl::range<1>(1));
      evaluator = std::make_shared<CPUEvaluator>(*mem_pool_buffer);
    }
    BOOST_TEST_MESSAGE("setup integration test fixture");
  }

  ~Fixture() { BOOST_TEST_MESSAGE("teardown integration test fixture"); }

  static std::shared_ptr<PortableMemPool> mem_pool_data;
  static std::shared_ptr<cl::sycl::buffer<PortableMemPool>> mem_pool_buffer;
  static std::shared_ptr<CPUEvaluator> evaluator;
};

std::shared_ptr<PortableMemPool> Fixture::mem_pool_data = nullptr;
std::shared_ptr<cl::sycl::buffer<PortableMemPool>> Fixture::mem_pool_buffer =
    nullptr;
std::shared_ptr<CPUEvaluator> Fixture::evaluator = nullptr;

namespace {

Index_t get_alloc_count(cl::sycl::buffer<PortableMemPool> mem_pool_buffer) {
  auto host_acc = mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
  return host_acc[0].get_total_allocation_count();
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
run_program(const std::string &path,
           const std::shared_ptr<CPUEvaluator> &evaluator,
           cl::sycl::buffer<PortableMemPool> mem_pool_buffer,
           const float expected_val) {
  std::cout << std::endl;
  std::cout << "Running program " << path << std::endl;

  const auto initial_alloc_count = get_alloc_count(mem_pool_buffer);

  Parser parser(path);
  auto parsed_result = parser.parse_program();
  std::cout << "Parsed program: ";
  parsed_result->debug_print(0);
  std::cout << std::endl;

  Compiler compiler(parsed_result, mem_pool_buffer);
  auto compiled_result = compiler.compile();

  std::cout << "Compiled program: " << std::endl;
  compiler.debug_print_ast(compiled_result);
  std::cout << std::endl;

  const auto check_eval_program = [&](const auto program) {
    Index_t max_blocks_for_exec;
    const auto alloc_count_pre_eval = get_alloc_count(mem_pool_buffer);
    const auto program_result =
        evaluator->evaluate_program(compiled_result, max_blocks_for_exec);
    const auto alloc_count_post_eval = get_alloc_count(mem_pool_buffer);
    BOOST_CHECK_EQUAL(alloc_count_pre_eval, alloc_count_post_eval);
    std::cout << "Max concurrent blocks during exec: " << max_blocks_for_exec
              << std::endl;
    return program_result;
  };

  const auto program_result = check_eval_program(compiled_result);

  BlockPrep block_prep(64, 32, 32, mem_pool_buffer);
  compiled_result = block_prep.prepare_for_block_generation(compiled_result);
  std::cout << "Program after prep for block generation: " << std::endl;
  compiler.debug_print_ast(compiled_result);
  std::cout << std::endl;
  const auto compiled_program_result = check_eval_program(compiled_result);

  compiler.deallocate_ast(compiled_result);

  const auto final_alloc_count = get_alloc_count(mem_pool_buffer);
  BOOST_CHECK_EQUAL(initial_alloc_count, final_alloc_count);
  BOOST_CHECK_EQUAL(expected_val, program_result.m_data.float_val);
  BOOST_CHECK_EQUAL(expected_val, compiled_program_result.m_data.float_val);
  return compiled_program_result;
}
} // namespace

BOOST_FIXTURE_TEST_SUITE(IntegrationTests, Fixture)

// TODO memory leak in this one, fix it before enabling
/*
BOOST_AUTO_TEST_CASE(NumericIntegration) {
  run_program("../TestPrograms/NumericIntegration.fgpu", evaluator, *memPoolBuff, -1.18747);
}
*/

BOOST_AUTO_TEST_CASE(MultiLet) {
  const auto programResult =
      run_program("../TestPrograms/MultiLet.fgpu", evaluator, *mem_pool_buffer, 1);
}

BOOST_AUTO_TEST_CASE(NoBindings) {
  const auto programResult =
      run_program("../TestPrograms/NoBindings.fgpu", evaluator, *mem_pool_buffer, 14);
}

BOOST_AUTO_TEST_CASE(SimpleCall) {
  const auto programResult =
      run_program("../TestPrograms/SimpleCall.fgpu", evaluator, *mem_pool_buffer, 42);
}

BOOST_AUTO_TEST_CASE(CallMultiBindings) {
  const auto programResult = run_program(
      "../TestPrograms/CallMultiBindings.fgpu", evaluator, *mem_pool_buffer, 13);
}

BOOST_AUTO_TEST_CASE(ListExample) {
  const auto programResult =
      run_program("../TestPrograms/ListExample.fgpu", evaluator, *mem_pool_buffer, 6);
}

BOOST_AUTO_TEST_CASE(MultiList) {
  const auto programResult =
      run_program("../TestPrograms/MultiList.fgpu", evaluator, *mem_pool_buffer, 3);
}

BOOST_AUTO_TEST_CASE(MultiLetRec) {
  const auto programResult =
      run_program("../TestPrograms/MultiLetRec.fgpu", evaluator, *mem_pool_buffer, 135);
}

BOOST_AUTO_TEST_CASE(MergeSort) {
  const auto programResult =
      run_program("../TestPrograms/MergeSort.fgpu", evaluator, *mem_pool_buffer, 123456);
}

BOOST_AUTO_TEST_CASE(GraphColoring) {
  const auto programResult =
      run_program("../TestPrograms/GraphColoring.fgpu", evaluator, *mem_pool_buffer, 8);
}

BOOST_AUTO_TEST_CASE(ComplexBindings) {
  run_program("../TestPrograms/ComplexBindings.fgpu", evaluator, *mem_pool_buffer, 13);
}

BOOST_AUTO_TEST_CASE(SimpleLambda) {
  run_program("../TestPrograms/SimpleLambda.fgpu", evaluator, *mem_pool_buffer, 5);
}

BOOST_AUTO_TEST_CASE(SimpleLetRec) {
  run_program("../TestPrograms/SimpleLetRec.fgpu", evaluator, *mem_pool_buffer, 5);
}

BOOST_AUTO_TEST_CASE(MapExample) {
  run_program("../TestPrograms/MapExample.fgpu", evaluator, *mem_pool_buffer, 1);
}

BOOST_AUTO_TEST_CASE(CleanUpFixture) {
  evaluator = nullptr;
  mem_pool_buffer = nullptr;
  mem_pool_data = nullptr;
}

BOOST_AUTO_TEST_SUITE_END()
