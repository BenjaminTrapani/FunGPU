#define BOOST_TEST_MODULE EvaluatorTestModule

#include "core/evaluator_v2/evaluator.hpp"
#include "core/compiler.hpp"
#include "core/evaluator_v2/compile_program.hpp"
#include "core/evaluator_v2/instruction.hpp"
#include "core/parser.hpp"
#include "core/serialize_ast_as_fgpu_program.hpp"
#include "core/temporary_directory.h"
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <filesystem>
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

  void check_roundtripped_program_yields_same_result(
      const Float_t expected_val, const std::string &program_path) {
    Parser parser(program_path);
    auto parsed_result = parser.parse_program();

    Compiler compiler(parsed_result, mem_pool_buffer);
    const auto [compiled_result, all_identifiers] = compiler.compile();
    const auto temp_path = std::filesystem::temp_directory_path() /
                           std::filesystem::path(program_path).filename();
    const TemporaryDirectory temp_dir;
    const auto updated_program_path =
        temp_dir.path() / std::filesystem::path(program_path).filename();
    std::fstream fgpu_program_file(updated_program_path, std::ios::out);
    const auto fgpu_program = serialize_ast_as_fgpu_program(
        compiled_result, all_identifiers, mem_pool_buffer);
    fgpu_program_file << fgpu_program;
    fgpu_program_file.close();
    compiler.deallocate_ast(compiled_result);
    check_program_yields_result(expected_val, program_path);
    check_program_yields_result(expected_val, updated_program_path);
  }
};

namespace {
BOOST_FIXTURE_TEST_CASE(NoBindings, Fixture) {
  check_program_yields_result(14, "./test_programs/NoBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(ComplexBindings, Fixture) {
  check_program_yields_result(13, "./test_programs/ComplexBindings.fgpu");
}

BOOST_FIXTURE_TEST_CASE(SimpleCall, Fixture) {
  check_program_yields_result(42, "./test_programs/SimpleCall.fgpu");
}

BOOST_FIXTURE_TEST_CASE(SimpleLetRec, Fixture) {
  check_program_yields_result(5, "./test_programs/SimpleLetRec.fgpu");
}

BOOST_FIXTURE_TEST_CASE(MultiLetRec, Fixture) {
  check_program_yields_result(135, "./test_programs/MultiLetRec.fgpu");
}

BOOST_FIXTURE_TEST_CASE(ListExample, Fixture) {
  check_program_yields_result(6, "./test_programs/ListExample.fgpu");
}

BOOST_FIXTURE_TEST_CASE(MultiList, Fixture) {
  check_program_yields_result(3, "./test_programs/MultiList.fgpu");
}

BOOST_FIXTURE_TEST_CASE(MapExample, Fixture) {
  check_roundtripped_program_yields_same_result(
      1, "./test_programs/MapExample.fgpu");
}

BOOST_FIXTURE_TEST_CASE(MergeSort, Fixture) {
  check_roundtripped_program_yields_same_result(
      123456, "./test_programs/MergeSort.fgpu");
}

BOOST_FIXTURE_TEST_CASE(BranchInBinding, Fixture) {
  check_program_yields_result(130, "./test_programs/BranchInBinding.fgpu");
}

BOOST_FIXTURE_TEST_CASE(ConditionalLetRec, Fixture) {
  check_program_yields_result(120, "./test_programs/ConditionalLetRec.fgpu");
}

BOOST_FIXTURE_TEST_CASE(ExplicitFactorial, Fixture) {
  check_program_yields_result(120, "./test_programs/ExplicitFactorial.fgpu");
}

BOOST_FIXTURE_TEST_CASE(ConcurrentTransformEnsureExcessBlocksBuffered,
                        Fixture) {
  check_program_yields_result(134225920,
                              "./test_programs/ConcurrentTransform.fgpu");
}
} // namespace
} // namespace FunGPU::EvaluatorV2
