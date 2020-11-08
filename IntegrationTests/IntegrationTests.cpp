//
// Created by benjamintrapani on 7/31/19.
//

#define BOOST_TEST_MODULE IntegrationTestsModule
#include <boost/test/included/unit_test.hpp>

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
    if (memPoolData == nullptr) {
      memPoolData = std::make_shared<PortableMemPool>();
      memPoolBuff = std::make_shared<cl::sycl::buffer<PortableMemPool>>(
          memPoolData, cl::sycl::range<1>(1));
      evaluator = std::make_shared<CPUEvaluator>(*memPoolBuff);
    }
    BOOST_TEST_MESSAGE("setup integration test fixture");
  }

  ~Fixture() { BOOST_TEST_MESSAGE("teardown integration test fixture"); }

  static std::shared_ptr<PortableMemPool> memPoolData;
  static std::shared_ptr<cl::sycl::buffer<PortableMemPool>> memPoolBuff;
  static std::shared_ptr<CPUEvaluator> evaluator;
};

std::shared_ptr<PortableMemPool> Fixture::memPoolData = nullptr;
std::shared_ptr<cl::sycl::buffer<PortableMemPool>> Fixture::memPoolBuff =
    nullptr;
std::shared_ptr<CPUEvaluator> Fixture::evaluator = nullptr;

namespace {

Index_t GetAllocCount(cl::sycl::buffer<PortableMemPool> memPoolBuff) {
  auto hostAcc = memPoolBuff.get_access<cl::sycl::access::mode::read_write>();
  return hostAcc[0].GetTotalAllocationCount();
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
RunProgram(const std::string &path,
           const std::shared_ptr<CPUEvaluator> &evaluator,
           cl::sycl::buffer<PortableMemPool> memPoolBuff,
           const float expectedVal) {
  std::cout << std::endl;
  std::cout << "Running program " << path << std::endl;

  const auto initialAllocCount = GetAllocCount(memPoolBuff);

  Parser parser(path);
  auto parsedResult = parser.ParseProgram();
  std::cout << "Parsed program: ";
  parsedResult->DebugPrint(0);
  std::cout << std::endl;

  Compiler compiler(parsedResult, memPoolBuff);
  auto compiledResult = compiler.Compile();

  std::cout << "Compiled program: " << std::endl;
  compiler.DebugPrintAST(compiledResult);
  std::cout << std::endl;

  const auto checkEvalProgram = [&](const auto program) {
    Index_t maxBlocksForExec;
    const auto allocCountPreEval = GetAllocCount(memPoolBuff);
    const auto programResult =
      evaluator->EvaluateProgram(compiledResult, maxBlocksForExec);
    const auto allocCountPostEval = GetAllocCount(memPoolBuff);
    BOOST_CHECK_EQUAL(allocCountPreEval, allocCountPostEval);
    std::cout << "Max concurrent blocks during exec: " << maxBlocksForExec
              << std::endl;
    return programResult;
  };

  const auto programResult = checkEvalProgram(compiledResult);

  BlockPrep blockPrep(64, 32, 32, memPoolBuff);
  compiledResult = blockPrep.PrepareForBlockGeneration(compiledResult);
  std::cout << "Program after prep for block generation: " << std::endl;
  compiler.DebugPrintAST(compiledResult);
  std::cout << std::endl;
  const auto compiledProgramResult = checkEvalProgram(compiledResult);

  compiler.DeallocateAST(compiledResult);

  const auto finalAllocCount = GetAllocCount(memPoolBuff);
  BOOST_CHECK_EQUAL(initialAllocCount, finalAllocCount);
  BOOST_CHECK_EQUAL(expectedVal, programResult.m_data.floatVal);
  BOOST_CHECK_EQUAL(expectedVal, compiledProgramResult.m_data.floatVal);
  return programResult;
}
} // namespace

BOOST_FIXTURE_TEST_SUITE(IntegrationTests, Fixture)

// TODO memory leak in this one, fix it before enabling
/*
BOOST_AUTO_TEST_CASE(NumericIntegration) {
  RunProgram("../TestPrograms/NumericIntegration.fgpu", evaluator, *memPoolBuff, -1.18747);
}
*/

BOOST_AUTO_TEST_CASE(MultiLet) {
  const auto programResult =
      RunProgram("../TestPrograms/MultiLet.fgpu", evaluator, *memPoolBuff, 1);
}

BOOST_AUTO_TEST_CASE(NoBindings) {
  const auto programResult =
      RunProgram("../TestPrograms/NoBindings.fgpu", evaluator, *memPoolBuff, 14);
}

BOOST_AUTO_TEST_CASE(SimpleCall) {
  const auto programResult =
      RunProgram("../TestPrograms/SimpleCall.fgpu", evaluator, *memPoolBuff, 42);
}

BOOST_AUTO_TEST_CASE(CallMultiBindings) {
  const auto programResult = RunProgram(
      "../TestPrograms/CallMultiBindings.fgpu", evaluator, *memPoolBuff, 13);
}

BOOST_AUTO_TEST_CASE(ListExample) {
  const auto programResult =
      RunProgram("../TestPrograms/ListExample.fgpu", evaluator, *memPoolBuff, 6);
}

BOOST_AUTO_TEST_CASE(MultiList) {
  const auto programResult =
      RunProgram("../TestPrograms/MultiList.fgpu", evaluator, *memPoolBuff, 3);
}

BOOST_AUTO_TEST_CASE(MultiLetRec) {
  const auto programResult =
      RunProgram("../TestPrograms/MultiLetRec.fgpu", evaluator, *memPoolBuff, 135);
}

BOOST_AUTO_TEST_CASE(MergeSort) {
  const auto programResult =
      RunProgram("../TestPrograms/MergeSort.fgpu", evaluator, *memPoolBuff, 123456);
}

BOOST_AUTO_TEST_CASE(GraphColoring) {
  const auto programResult =
      RunProgram("../TestPrograms/GraphColoring.fgpu", evaluator, *memPoolBuff, 8);
}

BOOST_AUTO_TEST_CASE(ComplexBindings) {
  const auto programResult = RunProgram("../TestPrograms/ComplexBindings.fgpu", evaluator, *memPoolBuff, 13);
}

BOOST_AUTO_TEST_CASE(CleanUpFixture) {
  evaluator = nullptr;
  memPoolBuff = nullptr;
  memPoolData = nullptr;
}

BOOST_AUTO_TEST_SUITE_END()
