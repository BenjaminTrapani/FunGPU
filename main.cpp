#include "core/block_prep.hpp"
#include "core/compiler.hpp"
#include "core/cpu_evaluator.hpp"
#include "core/parser.hpp"
#include "core/portable_mem_pool.hpp"
#include <iostream>

using namespace FunGPU;

int main(int argc, char **argv) {
  auto mem_pool = std::make_shared<PortableMemPool>();
  try {
    cl::sycl::buffer<PortableMemPool> mem_pool_buff(mem_pool,
                                                    cl::sycl::range<1>(1));
    CPUEvaluator evaluator(mem_pool_buff);
    Index_t argv_index = 1;
    BlockPrep block_prep(64, 32, 32, mem_pool_buff);
    while (true) {
      const auto program_path = [&]() -> std::optional<std::string> {
        if (argv_index < argc) {
          return std::string(argv[argv_index++]);
        }

        std::cout << "Program to run(or q to quit): ";
        std::string interactive_path;
        std::cin >> interactive_path;
        if (interactive_path == "q") {
          return std::optional<std::string>();
        }
        return interactive_path;
      }();
      if (!program_path) {
        break;
      }
      Parser parser(*program_path);
      auto parsed_result = parser.parse_program();

      Compiler compiler(parsed_result, mem_pool_buff);
      Compiler::ASTNodeHandle compiled_result;
      try {
        compiled_result = compiler.compile().ast_root;
      } catch (const Compiler::CompileException &e) {
        std::cerr << "Failed to compile " << *program_path << ": " << e.what()
                  << std::endl;
        continue;
      }
      std::cout << "Original compilation without any modifications: "
                << std::endl;
      compiler.debug_print_ast(compiled_result);
      std::cout << std::endl;
      std::cout << "Updated for block generation: " << std::endl;
      compiled_result =
          block_prep.prepare_for_block_generation(compiled_result);
      compiler.debug_print_ast(compiled_result);
      std::cout << "Successfully compiled program " << *program_path
                << std::endl;
      Index_t max_concurrent_block_count;
      const auto program_result = evaluator.evaluate_program(
          compiled_result, max_concurrent_block_count);
      std::cout << program_result.m_data.float_val << std::endl;
      std::cout << "Max concurrent blocks: " << max_concurrent_block_count
                << std::endl;
      compiler.deallocate_ast(compiled_result);
    }
  } catch (const cl::sycl::exception &e) {
    std::cerr << "Sycl exception in main: " << e.what() << std::endl;
  }

  return 0;
}
