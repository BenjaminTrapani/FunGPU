#define BOOST_TEST_MODULE EvaluatorTestModule
#include "Core/EvaluatorV2/Evaluator.hpp"
#include "Core/EvaluatorV2/CompileProgram.hpp"
#include <boost/test/unit_test.hpp>

namespace FunGPU::EvaluatorV2 {
struct Fixture {
  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  Evaluator evaluator{mem_pool_buffer};

  void check_program_yields_result(const Float_t expected_val,
                                   const std::string &program_path) {
    const auto compiled_program =
        compile_program(program_path, Evaluator::REGISTERS_PER_THREAD,
                        Evaluator::THREADS_PER_BLOCK, mem_pool_buffer);
    const auto result = evaluator.compute(compiled_program);
    BOOST_CHECK_EQUAL(expected_val, result.data.float_val);
  }
};

namespace {
BOOST_FIXTURE_TEST_CASE(NoBindings, Fixture) {
  check_program_yields_result(14, "./TestPrograms/NoBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(ComplexBindings, Fixture) {
  check_program_yields_result(13, "./TestPrograms/ComplexBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(SimpleCall, Fixture) {
  check_program_yields_result(42, "./TestPrograms/SimpleCall.fgpu");
}

BOOST_FIXTURE_TEST_CASE(SimpleLetRec, Fixture) {
  check_program_yields_result(5, "./TestPrograms/SimpleLetRec.fgpu");
}
BOOST_FIXTURE_TEST_CASE(MultiLetRec, Fixture) {
  check_program_yields_result(135, "./TestPrograms/MultiLetRec.fgpu");
}
BOOST_FIXTURE_TEST_CASE(ListExample, Fixture) {
  check_program_yields_result(6, "./TestPrograms/ListExample.fgpu");
}
BOOST_FIXTURE_TEST_CASE(MultiList, Fixture) {
  check_program_yields_result(3, "./TestPrograms/MultiList.fgpu");
}
BOOST_FIXTURE_TEST_CASE(MapExample, Fixture) {
  check_program_yields_result(1, "./TestPrograms/MapExample.fgpu");
}
BOOST_FIXTURE_TEST_CASE(MergeSort, Fixture) {
  check_program_yields_result(123456, "./TestPrograms/MergeSort.fgpu");
}
} // namespace
} // namespace FunGPU::EvaluatorV2
