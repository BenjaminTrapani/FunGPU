#include <cstdint>

namespace FunGPU {
#pragma once

class Error {
public:
  enum class Type : std::uint8_t {
    Success = 0,
    InvalidType = 1,
    ArityMismatch = 2,
    InvalidIndex = 3,
    GCOutOfSlots = 4,
    EvaluatorOutOfActiveBlocks = 5,
    EvaluatorOutOfDeletionBlocks = 6,
    MemPoolAllocFailure = 7,
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
