#include "Core/Parser.h"
#include "Core/Compiler.h"
#include "Core/CPUEvaluator.h"
#include "Core/PortableMemPool.h"
#include <iostream>

using namespace FunGPU;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Expected paths to fgpus programs to run, exiting";
        return 1;
    }

    auto memPool = std::make_shared<PortableMemPool>();
    try
    {
        cl::sycl::buffer<PortableMemPool> memPoolBuff(memPool, cl::sycl::range<1>(1));
        CPUEvaluator evaluator(memPoolBuff);
        for (int i = 1; i < argc; ++i)
        {
            Parser parser((std::string(argv[i])));
            auto parsedResult = parser.ParseProgram();

            Compiler compiler(parsedResult, memPoolBuff);
            auto compiledResult = compiler.Compile();

            unsigned int maxConcurrentBlockCount;
            const auto programResult = evaluator.EvaluateProgram(compiledResult, maxConcurrentBlockCount);
            std::cout << programResult.m_data.doubleVal << std::endl;

            compiler.DeallocateAST(compiledResult);
        }
    } catch (cl::sycl::exception e) {
        std::cerr << "Sycl exception in main: " << e.what() << std::endl;
    }

	return 0;
}
