#define BOOST_TEST_MODULE BlockGeneratorTestsModule
#include <boost/test/included/unit_test.hpp>

#include "Core/EvaluatorV2/BlockGenerator.h"
#include "Core/BlockPrep.hpp"
#include "Core/Parser.hpp"
#include "Core/Compiler.hpp"
#include "Core/Parser.hpp"

namespace FunGPU::EvaluatorV2 {
  namespace {
    constexpr Index_t REGISTERS_PER_BLOCK = 64;

    struct Fixture {
      std::shared_ptr<PortableMemPool> mem_pool_data = std::make_shared<PortableMemPool>();
      cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data, cl::sycl::range<1>(1)};
      BlockPrep block_prep{REGISTERS_PER_BLOCK, 32, 32, mem_pool_buffer};
      BlockGenerator block_generator{mem_pool_buffer, REGISTERS_PER_BLOCK};
    };

    BOOST_FIXTURE_TEST_CASE(NoBindingsGeneratesOneBlock, Fixture) {
      Parser parser("../TestPrograms/NoBindings.fgpu");
      auto parsedResult = parser.ParseProgram();
      std::cout << "Parsed program: " << std::endl;
      parsedResult->DebugPrint(0);
      std::cout << std::endl;

      Compiler compiler(parsedResult, mem_pool_buffer);
      auto compiledResult = compiler.Compile();
      std::cout << "Compiled program: " << std::endl;
      compiler.DebugPrintAST(compiledResult);
      std::cout << std::endl;

      compiledResult = block_prep.PrepareForBlockGeneration(compiledResult);
      std::cout << "Program after prep for block generation: " << std::endl;
      compiler.DebugPrintAST(compiledResult);
      std::cout << std::endl;
    }
  }
}
