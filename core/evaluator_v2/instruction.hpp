#pragma once

#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"
#include <cstdint>
#include <string>
#include <type_traits>

namespace FunGPU::EvaluatorV2 {
enum class InstructionType : std::uint8_t {
  CREATE_LAMBDA,
  ASSIGN_CONSTANT,
  ASSIGN,
  CALL_INDIRECT,
  BLOCKING_CALL_INDIRECT,
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
  INSTRUCTION_BARRIER
};

struct CreateLambda {
  static constexpr InstructionType TYPE = InstructionType::CREATE_LAMBDA;
  bool equals(const CreateLambda &, PortableMemPool::HostAccessor_t &) const;

  Index_t target_register;
  Index_t block_idx;
  PortableMemPool::TrivialArrayHandle<Index_t> captured_indices;
};

struct AssignConstant {
  static constexpr InstructionType TYPE = InstructionType::ASSIGN_CONSTANT;
  bool equals(const AssignConstant &, PortableMemPool::HostAccessor_t &) const;

  Index_t target_register;
  Float_t constant;
};

struct Assign {
  static constexpr InstructionType TYPE = InstructionType::ASSIGN;
  bool equals(const Assign &, PortableMemPool::HostAccessor_t &) const;

  Index_t target_register;
  Index_t source_register;
};

struct CallIndirectCommon {
  Index_t target_register;
  Index_t lambda_idx;
  PortableMemPool::TrivialArrayHandle<Index_t> arg_indices;
};

struct CallIndirect : public CallIndirectCommon {
  static constexpr InstructionType TYPE = InstructionType::CALL_INDIRECT;
  bool equals(const CallIndirect &, PortableMemPool::HostAccessor_t &) const;
};

struct BlockingCallIndirect : public CallIndirectCommon {
  static constexpr InstructionType TYPE =
      InstructionType::BLOCKING_CALL_INDIRECT;
  bool equals(const BlockingCallIndirect &,
              PortableMemPool::HostAccessor_t &) const;
};

static_assert(std::is_standard_layout<CallIndirect>::value);
static_assert(std::is_standard_layout<BlockingCallIndirect>::value);

struct If {
  static constexpr InstructionType TYPE = InstructionType::IF;
  bool equals(const If &, PortableMemPool::HostAccessor_t &) const;

  Index_t predicate;
  Index_t goto_true;
  Index_t goto_false;
};

struct Floor {
  static constexpr InstructionType TYPE = InstructionType::FLOOR;
  bool equals(const Floor &, PortableMemPool::HostAccessor_t &) const;

  Index_t target_register;
  Index_t arg;
};

struct InstructionBarrier {
  static constexpr InstructionType TYPE = InstructionType::INSTRUCTION_BARRIER;
  bool equals(const InstructionBarrier &,
              PortableMemPool::HostAccessor_t &) const {
    return true;
  }
};

template <InstructionType TheType> struct BinaryOp {
  static constexpr InstructionType TYPE = TheType;

  bool equals(const BinaryOp &other, PortableMemPool::HostAccessor_t &) const {
    return target_register == other.target_register && lhs == other.lhs &&
           rhs == other.rhs;
  }

  Index_t target_register;
  Index_t lhs;
  Index_t rhs;
};

using Add = BinaryOp<InstructionType::ADD>;
using Sub = BinaryOp<InstructionType::SUB>;
using Mul = BinaryOp<InstructionType::MUL>;
using Div = BinaryOp<InstructionType::DIV>;
using Equal = BinaryOp<InstructionType::EQUAL>;
using GreaterThan = BinaryOp<InstructionType::GREATER_THAN>;
using Remainder = BinaryOp<InstructionType::REMAINDER>;
using Expt = BinaryOp<InstructionType::EXPT>;

template <typename T>
concept HasTargetRegister = requires(T x) { x.target_register; };

struct Instruction {
  std::string print(PortableMemPool::HostAccessor_t mem_pool_acc) const;
  bool equals(const Instruction &, PortableMemPool::HostAccessor_t &) const;
  void deallocate(PortableMemPool::HostAccessor_t);

  Instruction() = default;
  Instruction(const InstructionType type) : type(type) {}

  union Data {
    CreateLambda create_lambda;
    AssignConstant assign_constant;
    Assign assign;
    CallIndirect call_indirect;
    BlockingCallIndirect blocking_call_indirect;
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
decltype(auto) visit(const Instruction &instruction, CB &&cb,
                     UnexpectedCB &&unexpected_cb) {
#define _(Type)                                                                \
  case Type::TYPE:                                                             \
    return cb(*reinterpret_cast<const Type *>(&instruction.data));

  switch (instruction.type) {
    _(CreateLambda)
    _(AssignConstant)
    _(Assign)
    _(CallIndirect)
    _(BlockingCallIndirect)
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
    _(InstructionBarrier)
  }

  return unexpected_cb(instruction);
#undef _
}

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(Instruction &instruction, CB &&cb,
                     UnexpectedCB &&unexpected_cb) {
  return visit(
      std::as_const(instruction),
      [&](const auto &elem) {
        return cb(const_cast<std::remove_cvref_t<decltype(elem)> &>(elem));
      },
      [&](const auto &elem) {
        return unexpected_cb(
            const_cast<std::remove_cvref_t<decltype(elem)> &>(elem));
      });
}
} // namespace FunGPU::EvaluatorV2