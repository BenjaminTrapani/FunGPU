#pragma once

#include "Core/EvaluatorV2/Instruction.h"
#include "Core/PortableMemPool.hpp"

namespace FunGPU::EvaluatorV2 {
  struct Lambda {
    Lambda() = default;
    explicit Lambda(const PortableMemPool::ArrayHandle<Instruction>& instructions) : instructions(instructions) {}

    PortableMemPool::ArrayHandle<Instruction> instructions;
  };
}
