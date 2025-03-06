#define BOOST_TEST_MODULE EvaluatorTestModule

#include "Core/EvaluatorV2/Evaluator.hpp"
#include "Core/EvaluatorV2/CompileProgram.hpp"
#include "Core/EvaluatorV2/Instruction.h"
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
struct Fixture {
  std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                    cl::sycl::range<1>(1)};
  Evaluator evaluator{mem_pool_buffer};

  Index_t get_alloc_count() {
    auto hostAcc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    return hostAcc[0].get_total_allocation_count();
  }

  bool program_contains_lambdas_with_captures(
      const PortableMemPool::ArrayHandle<Lambda> &program) {
    auto mem_pool_acc =
        mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    const auto *lambdas = mem_pool_acc[0].deref_handle(program);
    for (Index_t i = 0; i < program.get_count(); ++i) {
      const auto *instructions =
          mem_pool_acc[0].deref_handle(lambdas[i].instructions);
      for (Index_t j = 0; j < lambdas[i].instructions.get_count(); ++j) {
        const auto contains_captures = visit(
            instructions[j],
            Visitor{
                [&](const CreateLambda &create_lambda) {
                  return create_lambda.captured_indices.unpack().get_count() >
                         0;
                },
                [](const auto &) { return false; }},
            [](const auto &) -> bool {
              throw std::invalid_argument("Encountered unknown instruction");
            });
        if (contains_captures) {
          return true;
        }
      }
    }
    return false;
  }

  void check_program_yields_result(const Float_t expected_val,
                                   const std::string &program_path) {
    const auto initial_alloc_count = get_alloc_count();
    const auto compiled_program =
        compile_program(program_path, Evaluator::REGISTERS_PER_THREAD,
                        Evaluator::THREADS_PER_BLOCK, mem_pool_buffer);
    const auto program_contains_captures =
        program_contains_lambdas_with_captures(compiled_program);
    std::cout << "Program contains captures: " << program_contains_captures
              << std::endl;
    const auto pre_eval_alloc_count = get_alloc_count();
    const auto result = evaluator.compute(compiled_program);
    const auto post_eval_alloc_count = get_alloc_count();
    {
      auto mem_pool_acc =
          mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
      deallocate_program(compiled_program, mem_pool_acc);
    }
    BOOST_CHECK_EQUAL(expected_val, result.data.float_val);
    const auto final_alloc_count = get_alloc_count();
    // TODO assert no allocation delta once captured runtime values are garbage
    // collected.
    if (program_contains_captures) {
      BOOST_CHECK_GE(post_eval_alloc_count, pre_eval_alloc_count);
    } else {
      BOOST_CHECK_EQUAL(post_eval_alloc_count, pre_eval_alloc_count);
    }
    BOOST_CHECK_EQUAL(initial_alloc_count,
                      final_alloc_count -
                          (post_eval_alloc_count - pre_eval_alloc_count));
    std::cout << "initial_alloc_count=" << initial_alloc_count
              << ", final_alloc_count=" << final_alloc_count << std::endl;
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
BOOST_FIXTURE_TEST_CASE(BranchInBinding, Fixture) {
  check_program_yields_result(130, "./TestPrograms/BranchInBinding.fgpu");
}
BOOST_FIXTURE_TEST_CASE(ConditionalLetRec, Fixture) {
  check_program_yields_result(120, "./TestPrograms/ConditionalLetRec.fgpu");
}
BOOST_FIXTURE_TEST_CASE(ExplicitFactorial, Fixture) {
  check_program_yields_result(120, "./TestPrograms/ExplicitFactorial.fgpu");
}
} // namespace
} // namespace FunGPU::EvaluatorV2
