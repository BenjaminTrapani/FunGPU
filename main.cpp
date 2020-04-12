#include "Core/CPUEvaluator.hpp"
#include "Core/Compiler.hpp"
#include "Core/Parser.hpp"
#include "Core/PortableMemPool.hpp"
#include <iostream>

using namespace FunGPU;

int main(int argc, char **argv) {
  auto memPool = std::make_shared<PortableMemPool>();
  try {
    cl::sycl::buffer<PortableMemPool> memPoolBuff(memPool,
                                                  cl::sycl::range<1>(1));
    CPUEvaluator evaluator(memPoolBuff);
    Index_t argvIndex = 1;
    while (true) {
      const auto programPath = [&]() -> std::optional<std::string> {
        if (argvIndex < argc) {
          return std::string(argv[argvIndex++]);
        }

        std::cout << "Program to run(or q to quit): ";
        std::string interactivePath;
        std::cin >> interactivePath;
        if (interactivePath == "q") {
          return std::optional<std::string>();
        }
        return interactivePath;
      }();
      if (!programPath) {
        break;
      }
      Parser parser(*programPath);
      auto parsedResult = parser.ParseProgram();

      Compiler compiler(parsedResult, memPoolBuff);
      Compiler::ASTNodeHandle compiledResult;
      try {
        compiledResult = compiler.Compile();
      } catch (const Compiler::CompileException &e) {
        std::cerr << "Failed to compile " << *programPath << ": " << e.What()
                  << std::endl;
        continue;
      }
      std::cout << "Successfully compiled program " << *programPath
                << std::endl;
      Index_t maxConcurrentBlockCount;
      const auto programResult =
          evaluator.EvaluateProgram(compiledResult, maxConcurrentBlockCount);
      std::cout << programResult.m_data.floatVal << std::endl;
      std::cout << "Max concurrent blocks: " << maxConcurrentBlockCount
                << std::endl;
      compiler.DeallocateAST(compiledResult);
    }
  } catch (const cl::sycl::exception &e) {
    std::cerr << "Sycl exception in main: " << e.what() << std::endl;
  }

  return 0;
}
