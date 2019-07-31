#include "Parser.h"
#include "Compiler.h"
#include "CPUEvaluator.h"
#include "PortableMemPool.h"
#include <iostream>

using namespace FunGPU;

int main(int argc, char** argv)
{
    auto memPool = std::make_shared<PortableMemPool>();
    try
    {
        cl::sycl::buffer<PortableMemPool> memPoolBuff(memPool, cl::sycl::range<1>(1));
        CPUEvaluator evaluator(memPoolBuff);

        std::array<std::string, 2> programsToRun{"TestPrograms/TestInlineCalls.fgpu",
                                                 "TestPrograms/MergeSort.fgpu"};

        for (auto program: programsToRun) {
            Parser parser(program);
            auto parsedResult = parser.ParseProgram();
            parsedResult->DebugPrint(0);

            std::cout << std::endl;
            std::cout << std::endl;

            Compiler compiler(parsedResult, memPoolBuff);
            auto compiledResult = compiler.Compile();
            compiler.DebugPrintAST(compiledResult);

            std::cout << "Will evaluate program" << std::endl;
            const auto programResult = evaluator.EvaluateProgram(compiledResult);
            std::cout << "Program result: " << programResult.m_data.doubleVal << std::endl;

            compiler.DeallocateAST(compiledResult);
        }
    } catch (cl::sycl::exception e) {
        std::cerr << "Sycl exception in main: " << e.what() << std::endl;
    }

	std::cout << "Final allocation count: " << memPool->GetTotalAllocationCount() << std::endl;

	return 0;
}
