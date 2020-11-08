#pragma once

#include "Core/PortableMemPool.hpp"
#include "Core/Types.hpp"

namespace FunGPU::EvaluatorV2 {
struct RuntimeValue;

struct FunctionValue {
  Index_t block_idx;
  PortableMemPool::ArrayHandle<RuntimeValue> captures;
};

struct RuntimeValue {
  enum class Type { FLOAT, LAMBDA };
  union Data {
    Float_t float_val;
    FunctionValue function_val;
  };

  Type type;
  Data data;
};
} // namespace FunGPU::EvaluatorV2