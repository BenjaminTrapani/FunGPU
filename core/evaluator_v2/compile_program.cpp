#include "core/evaluator_v2/compile_program.hpp"
#include "core/block_prep.hpp"
#include "core/compiler.hpp"
#include "core/evaluator_v2/block_generator.hpp"
#include "core/parser.hpp"
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
Program compile_program(const std::string &path,
                        const Index_t registers_per_thread,
                        const Index_t threads_per_block,
                        cl::sycl::buffer<PortableMemPool> &mem_pool_buffer) {
  Parser parser(path);
  auto parsed_result = parser.parse_program();
  std::cout << "Parsed program: " << std::endl;
  parsed_result->debug_print(0);
  std::cout << std::endl;

  BlockPrep block_prep(registers_per_thread, threads_per_block,
                       threads_per_block, mem_pool_buffer);
  BlockGenerator block_generator(mem_pool_buffer, registers_per_thread);

  Compiler compiler(parsed_result, mem_pool_buffer);
  ASTNodeHandle compiled_result;
  try {
    compiled_result = compiler.compile().ast_root;
  } catch (const Compiler::CompileException &e) {
    std::cerr << "Failed to compile " << path << ": " << e.what() << std::endl;
    throw std::runtime_error("Compilation failure");
  }
  std::cout << "Compiled program: " << std::endl;
  compiler.debug_print_ast(compiled_result);
  std::cout << std::endl;

  compiled_result = block_prep.prepare_for_block_generation(compiled_result);
  std::cout << "Program after prep for block generation: " << std::endl;
  compiler.debug_print_ast(compiled_result);
  std::cout << std::endl << std::endl;

  const auto program = block_generator.construct_blocks(compiled_result);
  compiler.deallocate_ast(compiled_result);
  auto mem_pool_acc =
      mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
  std::cout << "Printed program: " << std::endl;
  std::cout << print(program, mem_pool_acc);
  return program;
}
} // namespace FunGPU::EvaluatorV2