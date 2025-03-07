#include "Core/EvaluatorV2/compile_program.hpp"
#include "Core/EvaluatorV2/evaluator.hpp"
#include "Core/portable_mem_pool.hpp"
#include <iostream>

using namespace FunGPU;
using namespace FunGPU::EvaluatorV2;

int main(int argc, char **argv) {
  auto mem_pool = std::make_shared<PortableMemPool>();
  try {
    cl::sycl::buffer<PortableMemPool> mem_pool_buffer(mem_pool,
                                                      cl::sycl::range<1>(1));
    Evaluator evaluator(mem_pool_buffer);
    Index_t argvIndex = 1;
    while (true) {
      const auto program_path = [&]() -> std::optional<std::string> {
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
      if (!program_path) {
        break;
      }
      const auto program =
          compile_program(*program_path, Evaluator::REGISTERS_PER_THREAD,
                          Evaluator::THREADS_PER_BLOCK, mem_pool_buffer);
      const auto result = evaluator.compute(program);
      std::cout << "Result: " << result.data.float_val << std::endl;
    }
  } catch (const cl::sycl::exception &e) {
    std::cerr << "Sycl exception in main: " << e.what() << std::endl;
  }
  return 0;
}
