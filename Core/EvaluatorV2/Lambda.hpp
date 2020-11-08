#pragma once

#include "Core/EvaluatorV2/Instruction.h"
#include "Core/PortableMemPool.hpp"
#include <string>

namespace FunGPU::EvaluatorV2 {
struct Lambda {
  Lambda() = default;
  explicit Lambda(const PortableMemPool::ArrayHandle<Instruction> &instructions)
      : instructions(instructions) {}
  std::string print(PortableMemPool::HostAccessor_t) const;

  PortableMemPool::ArrayHandle<Instruction> instructions;
};
} // namespace FunGPU::EvaluatorV2
