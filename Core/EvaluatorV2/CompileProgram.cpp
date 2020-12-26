#include "Core/EvaluatorV2/CompileProgram.hpp"
#include "Core/BlockPrep.hpp"
#include "Core/Compiler.hpp"
#include "Core/EvaluatorV2/BlockGenerator.h"
#include "Core/Parser.hpp"

namespace FunGPU::EvaluatorV2 {
Program compile_program(const std::string &path,
                        const Index_t registers_per_thread,
                        const Index_t threads_per_block,
                        cl::sycl::buffer<PortableMemPool> &mem_pool_buffer) {
  Parser parser(path);
  auto parsed_result = parser.ParseProgram();
  std::cout << "Parsed program: " << std::endl;
  parsed_result->DebugPrint(0);
  std::cout << std::endl;

  BlockPrep block_prep(registers_per_thread, threads_per_block,
                       threads_per_block, mem_pool_buffer);
  BlockGenerator block_generator(mem_pool_buffer, registers_per_thread);

  Compiler compiler(parsed_result, mem_pool_buffer);
  auto compiled_result = compiler.Compile();
  std::cout << "Compiled program: " << std::endl;
  compiler.DebugPrintAST(compiled_result);
  std::cout << std::endl;

  compiled_result = block_prep.PrepareForBlockGeneration(compiled_result);
  std::cout << "Program after prep for block generation: " << std::endl;
  compiler.DebugPrintAST(compiled_result);
  std::cout << std::endl << std::endl;

  const auto program = block_generator.construct_blocks(compiled_result);
  auto mem_pool_acc =
      mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
  std::cout << "Printed program: " << std::endl;
  std::cout << print(program, mem_pool_acc);
  return program;
}
} // namespace FunGPU::EvaluatorV2