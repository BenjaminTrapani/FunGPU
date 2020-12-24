#pragma once

#include "Core/PortableMemPool.hpp"
#include "Core/Types.hpp"

namespace FunGPU::EvaluatorV2 {
struct RuntimeValue;

struct FunctionValue {
  FunctionValue() = default;
  FunctionValue(
      const Index_t block_idx,
      const PortableMemPool::TrivialArrayHandle<RuntimeValue> captures)
      : block_idx(block_idx), captures(captures) {}

  Index_t block_idx;
  PortableMemPool::TrivialArrayHandle<RuntimeValue> captures;
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