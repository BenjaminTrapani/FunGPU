#include "Parser.h"
#include "Compiler.h"
#include "CPUEvaluator.h"
#include <iostream>

using namespace FunGPU;

int main(int argc, char** argv)
{
	Parser parser("FastGetNPrimeSum.fgpu");
	auto parsedResult = parser.ParseProgram();
	parsedResult->DebugPrint(0);

	std::cout << std::endl;
	std::cout << std::endl;

	Compiler compiler(parsedResult);
	auto compiledResult = compiler.Compile();
	compiler.DebugPrintAST(compiledResult);

	CPUEvaluator evaluator(compiledResult);
	const auto programResult = evaluator.EvaluateProgram();
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "Program result: " << programResult.m_data.doubleVal;

	return 0;
}