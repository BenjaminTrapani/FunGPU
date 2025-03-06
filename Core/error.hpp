#include <cstdint>

namespace FunGPU {
#pragma once

class Error {
public:
  enum class Type : std::uint8_t {
    Success = 0,
    InvalidArgType = 1,
    InvalidASTType = 2,
    ArityMismatch = 3,
    InvalidIndex = 4,
    GCOutOfSlots = 5,
    EvaluatorOutOfActiveBlocks = 6,
    EvaluatorOutOfDeletionBlocks = 7,
    MemPoolAllocFailure = 8,
  };

  Error() : node_type(Type::Success) {}
  Error(const Error &other) : node_type(other.node_type) {}
  Error(const Type type) : node_type(type) {}

  Type GetType() const { return node_type; }

private:
  Type node_type;
};

#define RETURN_IF_FAILURE(expr)                                                \
  {                                                                            \
    const auto __error = expr;                                                 \
    if (__error.GetType() != Error::Type::Success) {                           \
      return __error;                                                          \
    }                                                                          \
  }

} // namespace FunGPU
