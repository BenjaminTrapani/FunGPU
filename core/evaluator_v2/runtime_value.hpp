#pragma once

#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"

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
  union Data {
    explicit Data(const Float_t float_val) : float_val(float_val) {}
    explicit Data(const FunctionValue function_val)
        : function_val(function_val) {}
    Data() = default;

    Float_t float_val;
    FunctionValue function_val;
  };

  explicit RuntimeValue(const Float_t float_val) : data(float_val) {}
  explicit RuntimeValue(const FunctionValue function_val)
      : data(function_val) {}
  RuntimeValue() = default;

  Data data;
};
} // namespace FunGPU::EvaluatorV2