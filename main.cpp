#include "Parser.h"
#include "Compiler.h"
#include <iostream>

using namespace FunGPU;

int main(int argc, char** argv)
{
	Parser parser("TestProgram.fgpu");
	auto parsedResult = parser.ParseProgram();
	parsedResult->DebugPrint(0);

	std::cout << std::endl;
	std::cout << std::endl;

	Compiler compiler(parsedResult);
	auto compiledResult = compiler.Compile();
	compiler.DebugPrintAST(compiledResult);

	return 0;
}