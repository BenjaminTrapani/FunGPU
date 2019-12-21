#include "Core/CPUEvaluator.h"
#include "Core/Compiler.h"
#include "Core/Parser.h"
#include "Core/PortableMemPool.hpp"
#include <iostream>

using namespace FunGPU;

int main(int argc, char **argv) {
  auto memPool = std::make_shared<PortableMemPool>();
  try {
    cl::sycl::buffer<PortableMemPool> memPoolBuff(memPool,
                                                  cl::sycl::range<1>(1));
    CPUEvaluator evaluator(memPoolBuff);
    while (true) {
      std::cout << "Program to run(or q to quit): ";
      std::string programPath;
      std::cin >> programPath;
      if (programPath == "q") {
        break;
      }
      Parser parser(programPath);
      auto parsedResult = parser.ParseProgram();

      Compiler compiler(parsedResult, memPoolBuff);
      Compiler::ASTNodeHandle compiledResult;
      try {
        compiledResult = compiler.Compile();
      } catch (const Compiler::CompileException &e) {
        std::cerr << "Failed to compile " << programPath << ": " << e.What()
                  << std::endl;
        continue;
      }
      std::cout << "Successfully compiled program " << programPath << std::endl;
      Index_t maxConcurrentBlockCount;
      const auto programResult =
          evaluator.EvaluateProgram(compiledResult, maxConcurrentBlockCount);
      std::cout << programResult.m_data.floatVal << std::endl;

      compiler.DeallocateAST(compiledResult);
    }
  } catch (const cl::sycl::exception &e) {
    std::cerr << "Sycl exception in main: " << e.what() << std::endl;
  }

  return 0;
}
