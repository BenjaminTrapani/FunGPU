//
// Created by benjamintrapani on 7/31/19.
//

#define BOOST_TEST_MODULE IntegrationTestsModule
#include <boost/test/included/unit_test.hpp>

#include "CPUEvaluator.h"
#include "Compiler.h"
#include "Parser.h"
#include "PortableMemPool.hpp"

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

size_t GetAllocCount(cl::sycl::buffer<PortableMemPool> memPoolBuff) {
  auto hostAcc = memPoolBuff.get_access<cl::sycl::access::mode::read_write>();
  return hostAcc[0].GetTotalAllocationCount();
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
RunProgram(const std::string &path,
           const std::shared_ptr<CPUEvaluator> &evaluator,
           cl::sycl::buffer<PortableMemPool> memPoolBuff) {
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

  unsigned int maxBlocksForExec;
  const auto programResult =
      evaluator->EvaluateProgram(compiledResult, maxBlocksForExec);
  std::cout << "Max concurrent blocks during exec: " << maxBlocksForExec
            << std::endl;
  compiler.DeallocateAST(compiledResult);

  const auto finalAllocCount = GetAllocCount(memPoolBuff);
  BOOST_REQUIRE_EQUAL(initialAllocCount, finalAllocCount);

  return programResult;
}
}

BOOST_FIXTURE_TEST_SUITE(IntegrationTests, Fixture)

BOOST_AUTO_TEST_CASE(MultiLet) {
  const auto programResult =
      RunProgram("../TestPrograms/MultiLet.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(1, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(NoBindings) {
  const auto programResult =
      RunProgram("../TestPrograms/NoBindings.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(14, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(SimpleCall) {
  const auto programResult =
      RunProgram("../TestPrograms/SimpleCall.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(42, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(CallMultiBindings) {
  const auto programResult = RunProgram(
      "../TestPrograms/CallMultiBindings.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(13, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(ListExample) {
  const auto programResult =
      RunProgram("../TestPrograms/ListExample.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(6, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(MultiList) {
  const auto programResult =
      RunProgram("../TestPrograms/MultiList.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(3, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(MultiLetRec) {
  const auto programResult =
      RunProgram("../TestPrograms/MultiLetRec.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(135, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(MergeSort) {
  const auto programResult =
      RunProgram("../TestPrograms/MergeSort.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(123456, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(GraphColoring) {
  const auto programResult =
      RunProgram("../TestPrograms/GraphColoring.fgpu", evaluator, *memPoolBuff);
  BOOST_REQUIRE_EQUAL(8, programResult.m_data.doubleVal);
}

BOOST_AUTO_TEST_CASE(CleanUpFixture) {
  evaluator = nullptr;
  memPoolBuff = nullptr;
  memPoolData = nullptr;
}

BOOST_AUTO_TEST_SUITE_END()
