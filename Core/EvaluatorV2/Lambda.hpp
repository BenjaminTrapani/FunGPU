#pragma once

#include "Core/EvaluatorV2/Instruction.hpp"
#include "Core/PortableMemPool.hpp"
#include "Core/Types.hpp"
#include <string>

namespace FunGPU::EvaluatorV2 {
struct Lambda {
  struct InstructionProperties {
    Index_t total_num_indirect_calls;
    PortableMemPool::ArrayHandle<Index_t> num_runtime_values_per_op;
  };

  Lambda() = default;
  explicit Lambda(const PortableMemPool::ArrayHandle<Instruction> &instructions,
                  PortableMemPool::HostAccessor_t &mem_pool_acc);
  std::string print(PortableMemPool::HostAccessor_t) const;
  void deallocate(PortableMemPool::HostAccessor_t);

  PortableMemPool::ArrayHandle<Instruction> instructions;
  InstructionProperties instruction_properties;
};
} // namespace FunGPU::EvaluatorV2
