#include "Core/EvaluatorV2/Lambda.hpp"
#include "Core/PortableMemPool.hpp"
#include "Core/Visitor.hpp"
#include <sstream>

namespace FunGPU::EvaluatorV2 {
namespace {
Lambda::InstructionProperties derive_instruction_properties(
    const PortableMemPool::ArrayHandle<Instruction> &instructions,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  Lambda::InstructionProperties result;
  const auto *instruction_data = mem_pool_acc[0].derefHandle(instructions);
  std::vector<Index_t> num_runtime_values_per_op;
  for (Index_t i = 0; i < instructions.GetCount(); ++i) {
    const auto &instruction = instruction_data[i];
    visit(instruction,
          Visitor{
              [&](const CallIndirect &call_indirect) {
                num_runtime_values_per_op.emplace_back(
                    call_indirect.arg_indices.unpack().GetCount());
                ++result.total_num_indirect_calls;
              },
              [&](const BlockingCallIndirect &blocking_call_indirect) {
                num_runtime_values_per_op.emplace_back(
                    blocking_call_indirect.arg_indices.unpack().GetCount());
                ++result.total_num_indirect_calls;
              },
              [&](const CreateLambda &create_lambda) {
                num_runtime_values_per_op.emplace_back(
                    create_lambda.captured_indices.unpack().GetCount());
              },
              [](const auto &) {},
          },
          [](const auto &) {});
  }
  result.num_runtime_values_per_op =
      mem_pool_acc[0].AllocArray<Index_t>(num_runtime_values_per_op.size());
  auto *num_runtime_values_per_op_data =
      mem_pool_acc[0].derefHandle(result.num_runtime_values_per_op);
  for (Index_t i = 0; i < num_runtime_values_per_op.size(); ++i) {
    num_runtime_values_per_op_data[i] = num_runtime_values_per_op[i];
  }
  return result;
}
} // namespace

Lambda::Lambda(const PortableMemPool::ArrayHandle<Instruction> &instructions,
               PortableMemPool::HostAccessor_t &mem_pool_acc)
    : instructions(instructions),
      instruction_properties(
          derive_instruction_properties(instructions, mem_pool_acc)) {}

void Lambda::deallocate(PortableMemPool::HostAccessor_t mem_pool_acc) {
  auto *instruction_data = mem_pool_acc[0].derefHandle(instructions);
  for (Index_t i = 0; i < instructions.GetCount(); ++i) {
    instruction_data[i].deallocate(mem_pool_acc);
  }
  mem_pool_acc[0].DeallocArray(instructions);
  mem_pool_acc[0].DeallocArray(
      instruction_properties.num_runtime_values_per_op);
}

std::string Lambda::print(PortableMemPool::HostAccessor_t mem_pool_acc) const {
  std::stringstream result;
  const auto *instruction_data = mem_pool_acc[0].derefHandle(instructions);
  for (Index_t i = 0; i < instructions.GetCount(); ++i) {
    result << i << ": " << instruction_data[i].print(mem_pool_acc) << std::endl;
  }
  return result.str();
}
} // namespace FunGPU::EvaluatorV2
