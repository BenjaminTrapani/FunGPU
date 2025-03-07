#include "core/evaluator_v2/program.hpp"
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
} // namespace FunGPU::EvaluatorV2
