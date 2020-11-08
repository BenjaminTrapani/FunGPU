#include "Core/EvaluatorV2/Lambda.hpp"
#include <sstream>

namespace FunGPU::EvaluatorV2 {
std::string Lambda::print(PortableMemPool::HostAccessor_t mem_pool_acc) const {
  std::stringstream result;
  const auto *instruction_data = mem_pool_acc[0].derefHandle(instructions);
  for (Index_t i = 0; i < instructions.GetCount(); ++i) {
    result << i << ": " << instruction_data[i].print(mem_pool_acc) << std::endl;
  }
  return result.str();
}
} // namespace FunGPU::EvaluatorV2
