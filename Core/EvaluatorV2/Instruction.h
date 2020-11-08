#pragma once

#include "Core/PortableMemPool.hpp"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/Types.hpp"
#include <type_traits>

namespace FunGPU::EvaluatorV2 {
enum class InstructionType {
  CREATE_LAMBDA,
  ASSIGN_CONSTANT,
  CALL_INDIRECT,
  IF,
  ADD,
  SUB,
  MUL,
  DIV,
  EQUAL,
  GREATER_THAN,
  REMAINDER,
  EXPT,
  FLOOR,
};

struct CreateLambda {
  static constexpr InstructionType TYPE = InstructionType::CREATE_LAMBDA;
  Index_t target_register;
  Index_t block_idx;
  PortableMemPool::TrivialArrayHandle<Index_t> captured_indices;
};

struct AssignConstant {
  static constexpr InstructionType TYPE = InstructionType::ASSIGN_CONSTANT;
  Index_t target_register;
  Float_t constant;
};

struct CallIndirect {
  static constexpr InstructionType TYPE = InstructionType::CALL_INDIRECT;
  Index_t target_register;
  Index_t lambda_idx;
  PortableMemPool::TrivialArrayHandle<Index_t> arg_indices;
};

struct If {
  static constexpr InstructionType TYPE = InstructionType::IF;
  Index_t predicate;
  Index_t goto_true;
  Index_t goto_false;
};

struct Floor {
  static constexpr InstructionType TYPE = InstructionType::FLOOR;
  Index_t arg;
  Index_t target_register;
};

template <InstructionType TheType> struct BinaryOp {
  static constexpr InstructionType TYPE = TheType;
  Index_t lhs;
  Index_t rhs;
  Index_t target_register;
};

using Add = BinaryOp<InstructionType::ADD>;
using Sub = BinaryOp<InstructionType::SUB>;
using Mul = BinaryOp<InstructionType::MUL>;
using Div = BinaryOp<InstructionType::DIV>;
using Equal = BinaryOp<InstructionType::EQUAL>;
using GreaterThan = BinaryOp<InstructionType::GREATER_THAN>;
using Remainder = BinaryOp<InstructionType::REMAINDER>;
using Expt = BinaryOp<InstructionType::EXPT>;

struct Instruction {
  union Data {
    CreateLambda create_lambda;
    AssignConstant assign_constant;
    CallIndirect call_indirect;
    If if_val; 
    Add add; 
    Sub sub;
    Mul mul;
    Div div;
    Equal equal; 
    GreaterThan greater_than; 
    Remainder remainder;
    Expt expt; 
    Floor floor;
  } data;

  InstructionType type;
};

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(const Instruction &instruction, CB &&cb, UnexpectedCB&& unexpected_cb) {
#define _(Type)                                                                \
  case Type::TYPE:                                                             \
    return cb(*reinterpret_cast<const Type*>(&instruction.data));

  switch (instruction.type) {
    _(CreateLambda)
    _(AssignConstant)
    _(CallIndirect)
    _(If)
    _(Add)
    _(Sub)
    _(Mul)
    _(Div)
    _(Equal)
    _(GreaterThan)
    _(Remainder)
    _(Expt)
    _(Floor)
  }

  unexpected_cb(instruction);
  #undef _
}

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(Instruction &instruction, CB &&cb, UnexpectedCB&& unexpected_cb) {
  return visit(std::as_const(instruction), [&](const auto& elem) {
    cb(const_cast<std::remove_const_t<decltype(elem)>>(elem));
  }, [&](const auto& elem) {
    unexpected_cb(const_cast<std::remove_const_t<decltype(elem)>>(elem));
  });
}
} // namespace FunGPU::EvaluatorV2