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

	const auto poolSize = static_cast<size_t>(pow(2, 29));
	auto memPool = std::make_shared<PortableMemPool>(std::vector<std::pair<size_t, size_t>>{ {8, poolSize}, { 64, poolSize }, 
		{ 512, poolSize }, { 8192, poolSize }});
	Compiler compiler(parsedResult, memPool);
	auto compiledResult = compiler.Compile();
	compiler.DebugPrintAST(compiledResult);

	auto memPoolCpy = std::make_shared<PortableMemPool>(*memPool);
	// Verify that we can move the mem pool without breaking outstanding references.
	CPUEvaluator evaluator(compiledResult, memPoolCpy);
	const auto programResult = evaluator.EvaluateProgram();
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "Program result: " << programResult.m_data.doubleVal;

	return 0;
}