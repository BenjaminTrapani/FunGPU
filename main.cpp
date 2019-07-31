#include "Parser.h"
#include "Compiler.h"
#include "CPUEvaluator.h"
#include "PortableMemPool.h"
#include <iostream>

using namespace FunGPU;

int main(int argc, char** argv)
{
	Parser parser("MergeSort.fgpu");
	auto parsedResult = parser.ParseProgram();
	parsedResult->DebugPrint(0);

	std::cout << std::endl;
	std::cout << std::endl;

	auto memPool = std::make_shared<PortableMemPool>();
	Compiler compiler(parsedResult, memPool);
	auto compiledResult = compiler.Compile();
	compiler.DebugPrintAST(compiledResult);

	//auto memPoolCpy = std::make_shared<PortableMemPool>(*memPool);
	// Verify that we can move the mem pool without breaking outstanding references.
	{
		CPUEvaluator evaluator(compiledResult, memPool);
		std::cout << "Will evaluate program" << std::endl;
		const auto programResult = evaluator.EvaluateProgram();
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "Program result: " << programResult.m_data.doubleVal << std::endl;
	}
	compiler.DeallocateAST(compiledResult);

	std::cout << "Final allocation count: " << memPool->GetTotalAllocationCount() << std::endl;

	return 0;
}
