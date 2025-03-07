#include "core/evaluator_v2/lambda.hpp"
#include "core/evaluator_v2/runtime_block.hpp"
#include "core/portable_mem_pool.hpp"
#include "core/visitor.hpp"
#include <sstream>

namespace FunGPU::EvaluatorV2 {
namespace {
Lambda::InstructionProperties derive_instruction_properties(
    const PortableMemPool::ArrayHandle<Instruction> &instructions,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  Lambda::InstructionProperties result;
  const auto *instruction_data = mem_pool_acc[0].deref_handle(instructions);
  std::vector<Index_t> num_runtime_values_per_op;
  for (Index_t i = 0; i < instructions.get_count(); ++i) {
    const auto &instruction = instruction_data[i];
    visit(instruction,
          Visitor{
              [&](const CallIndirect &call_indirect) {
                num_runtime_values_per_op.emplace_back(
                    call_indirect.arg_indices.unpack().get_count());
                ++result.total_num_indirect_calls;
              },
              [&](const BlockingCallIndirect &blocking_call_indirect) {
                num_runtime_values_per_op.emplace_back(
                    blocking_call_indirect.arg_indices.unpack().get_count());
                ++result.total_num_indirect_calls;
              },
              [&](const CreateLambda &create_lambda) {
                num_runtime_values_per_op.emplace_back(
                    create_lambda.captured_indices.unpack().get_count());
              },
              [](const auto &) {},
          },
          [](const auto &) {});
  }
  result.num_runtime_values_per_op =
      mem_pool_acc[0].alloc_array<Index_t>(num_runtime_values_per_op.size());
  auto *num_runtime_values_per_op_data =
      mem_pool_acc[0].deref_handle(result.num_runtime_values_per_op);
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
  auto *instruction_data = mem_pool_acc[0].deref_handle(instructions);
  for (Index_t i = 0; i < instructions.get_count(); ++i) {
    instruction_data[i].deallocate(mem_pool_acc);
  }
  mem_pool_acc[0].dealloc_array(instructions);
  mem_pool_acc[0].dealloc_array(
      instruction_properties.num_runtime_values_per_op);
}

std::string Lambda::print(PortableMemPool::HostAccessor_t mem_pool_acc) const {
  std::stringstream result;
  const auto *instruction_data = mem_pool_acc[0].deref_handle(instructions);
  Index_t pre_allocated_rv_index = 0U;
  const auto *pre_allocated_count = mem_pool_acc[0].deref_handle(
      instruction_properties.num_runtime_values_per_op);
  for (Index_t i = 0; i < instructions.get_count(); ++i) {
    result << i << ": " << instruction_data[i].print(mem_pool_acc);
    visit(instruction_data[i],
          Visitor{[&](const OneOf<CallIndirect, BlockingCallIndirect,
                                  CreateLambda> auto &) {
                    result << " (pre_allocated_rv_index: "
                           << pre_allocated_rv_index << ", num_allocated: "
                           << pre_allocated_count[pre_allocated_rv_index]
                           << ")";
                    ++pre_allocated_rv_index;
                  },
                  [](const auto &) {}},
          [](const auto &) {});
    result << std::endl;
  }
  return result.str();
}
} // namespace FunGPU::EvaluatorV2
