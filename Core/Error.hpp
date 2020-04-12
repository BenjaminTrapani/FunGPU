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

  Error() : m_type(Type::Success) {}
  Error(const Error &other) : m_type(other.m_type) {}
  Error(const Type type) : m_type(type) {}

  Type GetType() const { return m_type; }

private:
  Type m_type;
};

#define RETURN_IF_FAILURE(expr)                                                \
  {                                                                            \
    const auto __error = expr;                                                 \
    if (__error.GetType() != Error::Type::Success) {                           \
      return __error;                                                          \
    }                                                                          \
  }

} // namespace FunGPU
