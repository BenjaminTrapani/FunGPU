#include "core/evaluator_v2/program.hpp"
#include <span>
#include <sstream>
namespace FunGPU::EvaluatorV2 {
std::string print(const Program &program,
                  PortableMemPool::HostAccessor_t mem_pool_acc) {
  std::stringstream result;
  const auto *lambdas = mem_pool_acc[0].deref_handle(program);
  for (Index_t i = 0; i < program.get_count(); ++i) {
    result << "Lambda " << i << ": " << std::endl;
    result << lambdas[i].print(mem_pool_acc) << std::endl;
  }
  return result.str();
}

void deallocate_program(const Program &program,
                        PortableMemPool::HostAccessor_t mem_pool_acc) {
  auto *lambdas = mem_pool_acc[0].deref_handle(program);
  for (Index_t i = 0; i < program.get_count(); ++i) {
    auto &lambda = lambdas[i];
    lambda.deallocate(mem_pool_acc);
  }
  mem_pool_acc[0].dealloc_array(program);
}

Index_t max_num_instructions_in_program(
    const Program &program,
    PortableMemPool::HostReadOnlyAccessorType mem_pool_acc) {
  Index_t max_num_instructions = 0;
  for (const auto &lambda :
       std::span(mem_pool_acc[0].deref_handle(program), program.get_count())) {
    max_num_instructions =
        std::max(max_num_instructions, lambda.instructions.get_count());
  }
  return max_num_instructions;
}
} // namespace FunGPU::EvaluatorV2
