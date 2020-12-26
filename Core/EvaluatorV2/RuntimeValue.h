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
    explicit Data(const Float_t float_val) : float_val(float_val) {}
    explicit Data(const FunctionValue function_val)
        : function_val(function_val) {}
    Data() = default;

    Float_t float_val;
    FunctionValue function_val;
  };

  explicit RuntimeValue(const Float_t float_val)
      : type(Type::FLOAT), data(float_val) {}
  explicit RuntimeValue(const FunctionValue function_val)
      : type(Type::LAMBDA), data(function_val) {}
  RuntimeValue() = default;

  Type type;
  Data data;
};
} // namespace FunGPU::EvaluatorV2