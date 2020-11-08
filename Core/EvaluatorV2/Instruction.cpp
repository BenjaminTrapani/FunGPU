#include "Core/EvaluatorV2/Instruction.h"
#include "Core/Visitor.hpp"
#include <sstream>

namespace FunGPU::EvaluatorV2 {
std::string
Instruction::print(PortableMemPool::HostAccessor_t mem_pool_acc) const {
  std::stringstream result;
#define HANDLE_BINARY_OP(TYPE, OP)                                             \
  [&](const TYPE &elem) {                                                      \
    result << #TYPE << ": reg " << elem.target_register << " = " << elem.lhs   \
           << " " << #OP << " " << elem.rhs;                                   \
  }

  visit(
      *this,
      Visitor{
          [&](const CreateLambda &create_lambda) {
            result << "CreateLambda: reg " << create_lambda.target_register
                   << " = " << create_lambda.block_idx
                   << ", capture registers ";
            const auto *captured_indices = mem_pool_acc[0].derefHandle(
                create_lambda.captured_indices.unpack());
            for (Index_t i = 0;
                 i < create_lambda.captured_indices.unpack().GetCount(); ++i) {
              result << captured_indices[i] << ", ";
            }
          },
          [&](const AssignConstant &assign_constant) {
            result << "AssignConstant: reg " << assign_constant.target_register
                   << " = " << assign_constant.constant;
          },
          [&](const CallIndirect &call_indirect) {
            result << "CallIndirect: reg " << call_indirect.target_register
                   << " = call reg " << call_indirect.lambda_idx
                   << ", arg registers ";
            const auto *arg_regs =
                mem_pool_acc[0].derefHandle(call_indirect.arg_indices.unpack());
            for (Index_t i = 0;
                 i < call_indirect.arg_indices.unpack().GetCount(); ++i) {
              result << arg_regs[i] << ", ";
            }
          },
          [&](const If &if_inst) {
            result << "If reg " << if_inst.predicate << " goto "
                   << if_inst.goto_true << " else " << if_inst.goto_false;
          },
          [&](const Floor &floor_inst) {
            result << "Floor: reg " << floor_inst.target_register << " = foor("
                   << floor_inst.arg << ")";
          },
          HANDLE_BINARY_OP(Add, +),
          HANDLE_BINARY_OP(Sub, -),
          HANDLE_BINARY_OP(Mul, *),
          HANDLE_BINARY_OP(Div, /),
          HANDLE_BINARY_OP(Equal, ==),
          HANDLE_BINARY_OP(GreaterThan, >),
          HANDLE_BINARY_OP(Remainder, remainder),
          HANDLE_BINARY_OP(Expt, expt),
      },
      [](const auto &) {
        throw std::invalid_argument("Unexpected instruction type");
      });

  return result.str();
#undef HANDLE_BINARY_OP
}

namespace {
template <typename T>
bool array_handles_equal(const PortableMemPool::ArrayHandle<T> lhs,
                         const PortableMemPool::ArrayHandle<T> rhs,
                         PortableMemPool::HostAccessor_t &mem_pool_acc) {
  if (lhs.GetCount() != rhs.GetCount()) {
    return false;
  }

  const auto *lhs_data = mem_pool_acc[0].derefHandle(lhs);
  const auto *rhs_data = mem_pool_acc[0].derefHandle(rhs);
  return std::equal(lhs_data, lhs_data + lhs.GetCount(), rhs_data);
}
} // namespace

bool CreateLambda::equals(const CreateLambda &other,
                          PortableMemPool::HostAccessor_t &mem_pool_acc) const {
  return target_register == other.target_register &&
         block_idx == other.block_idx &&
         array_handles_equal(captured_indices.unpack(),
                             other.captured_indices.unpack(), mem_pool_acc);
}

bool AssignConstant::equals(
    const AssignConstant &other,
    PortableMemPool::HostAccessor_t &mem_pool_acc) const {
  return target_register == other.target_register && constant == other.constant;
}

bool CallIndirect::equals(const CallIndirect &other,
                          PortableMemPool::HostAccessor_t &mem_pool_acc) const {
  return target_register == other.target_register &&
         lambda_idx == other.lambda_idx &&
         array_handles_equal(arg_indices.unpack(), other.arg_indices.unpack(),
                             mem_pool_acc);
}

bool If::equals(const If &other,
                PortableMemPool::HostAccessor_t &mem_pool_acc) const {
  return predicate == other.predicate && goto_true == other.goto_true &&
         goto_false == other.goto_false;
}

bool Floor::equals(const Floor &other,
                   PortableMemPool::HostAccessor_t &mem_pool_acc) const {
  return target_register == other.target_register && arg == other.arg;
}

bool Instruction::equals(const Instruction &other,
                         PortableMemPool::HostAccessor_t &mem_pool_acc) const {
  if (type != other.type) {
    return false;
  }
  bool matches = false;
  visit(
      *this,
      [&](const auto &lhs) {
        visit(
            other,
            [&](const auto &rhs) {
              if constexpr (std::is_same_v<decltype(lhs), decltype(rhs)>) {
                matches = lhs.equals(rhs, mem_pool_acc);
              }
            },
            [](const auto &) {
              throw std::invalid_argument("Unknown instruction type in equals");
            });
      },
      [](const auto &) {
        throw std::invalid_argument("Unknown instruction type in equals");
      });
  return matches;
}
} // namespace FunGPU::EvaluatorV2