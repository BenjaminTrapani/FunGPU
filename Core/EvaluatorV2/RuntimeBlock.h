#pragma once

#include "Core/EvaluatorV2/Instruction.h"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/Types.hpp"
#include "Core/Visitor.hpp"
#include <cstdint>

namespace FunGPU::EvaluatorV2 {
template <Index_t InstructionsPerBlock, Index_t RegistersPerThread,
          Index_t ThreadsPerBlock>
class RuntimeBlock
    : public PortableMemPool::EnableHandleFromThis<RuntimeBlock<
          InstructionsPerBlock, RegistersPerThread, ThreadsPerBlock>> {
public:
  struct TargetAddress {
    PortableMemPool<RuntimeBlock> block;
    Index_t thread;
    Index_t register_idx;
  };

  template <typename OnIndirectCall> bool step(Index_t thread);

private:
  Index_t next_cycle[ThreadsPerBlock]{0};
  Instruction[InstructionsPerBlock] instructions;
  RuntimeValue registers[ThreadsPerBlock][RegistersPerThread];
  PortableMemPool<RuntimeBlock> m_handle;
  TargetAddress target_data[ThreadsPerBlock];
  int num_outstanding_dependencies = 0;
};

template <Index_t InstructionsPerBlock, Index_t RegistersPerThread,
          Index_t ThreadsPerBlock>
template <typename OnIndirectCall>
template <typename OnActiveBlock>
bool RuntimeBlock<InstructionsPerBlock, RegistersPerThread, ThreadsPerBlock>::
    step(const Index_t thread, PortableMemPool::DeviceAccessor_t mem_pool,
         const Index_t cur_cycle, OnIndirectCall &&on_indirect_call,
         OnActiveBlock &&on_activate_block) {
  auto &cycle = next_cycle[thread];
  if (cycle != cur_cycle) {
    return;
  }
  ++cycle;

  if ((cur_cycle == InstructionsPerBlock ||
       instructions[cur_cycle].type == InstructionType::NOOP)) {
    const auto &target = target_data[thread];
    auto &target_block = *mem_pool[0].derefHandle(target.block);
    target_block.fill_dependency(target.thread, target.register_idx,
                                 registers[last_write_location],
                                 on_activate_block);
    return;
  }

  const auto &instruction = instructions[cur_cycle];
  auto &register_set = registers[thread];

#define HANDLE_BINARY_OP(TYPE, OP)                                             \
  [&](const TYPE &type) {                                                      \
    auto &target_register = register_set[type.result];                         \
    target_register.type = RuntimeValue::Type::FLOAT;                          \
    target_register.value =                                                    \
        register_set[type.lhs].float_val##OP register_set[type.rhs].float_val; \
  }

  visit(instruction,
        Visitor{HANDLE_BINARY_OP(Add, +), HANDLE_BINARY_OP(Sub, -),
                HANDLE_BINARY_OP(Mul, *), HANDLE_BINARY_OP(Div, /),
                HANDLE_BINARY_OP(Equal, ==), HANDLE_BINARY_OP(GreaterThan, >),
                [&](const AssignConstant &assign_constant) {
                  last_write_location = assign_constant.target_register;
                  auto &target_register =
                      register_set[assign_constant.target_register];
                  target_register.type = RuntimeValue::Type::FLOAT;
                  target_register.value = assign_constant.constant;
                },
                [&](const CreateLambda &create_lambda) {
                  auto *captured_indices =
                      mem_pool[0].derefHandle(create_lambda.captured_indices);
                  last_write_location = create_lambda.target_register;
                  auto &target_register =
                      register_set[create_lambda.target_register];
                  auto captured_values = mem_pool[0].AllocArray<RuntimeValue>(
                      create_lambda.captured_indices.GetCount());
                  auto *captured_values_data =
                      mem_pool[0].derefHandle(captured_values);
                  for (Index_t i = 0;
                       i < create_lambda.captured_indices.GetCount(); ++i) {
                    captured_values_data[i] = register_set[captured_indices[i]];
                  }
                  target_register.type = RuntimeValue::Type::FUNCTION;
                  target_register.value =
                      FunctionValue(create_lambda.block_idx, captured_values);
                },
                [&](const CallIndirect &call_indirect) {
                  const auto *arg_indices =
                      mem_pool[0].derefHandle(call_indirect.arg_indices);
                  auto arg_values = mem_pool[0].AllocArray<RuntimeValue>(
                      call_indirect.arg_indices.GetCount());
                  auto *arg_values_data = mem_pool[0].derefHandle(arg_values);
                  for (Index_t i = 0; i < call_indirect.arg_indices.GetCount();
                       ++i) {
                    arg_values_data[i] = register_set[arg_values[i]];
                  }
                  cl::sycl::atomic<int> atomic_dep_count(
                      (cl::sycl::multi_ptr<
                          int, cl::sycl::access::address_space::global_space>(
                          num_outstanding_dependencies)));
                  atomic_dep_count.fetch_add(1);
                  on_indirect_call(m_handle, call_indirect.target_register,
                                   thread, arg_values);
                }},
        instruction);

#undef HANDLE_BINARY_OP
}

template <Index_t InstructionsPerBlock, Index_t RegistersPerThread,
          Index_t ThreadsPerBlock>
template <typename OnActiveBlock>
void RuntimeBlock<InstructionsPerBlock, RegistersPerThread, ThreadsPerBlock>::
    fill_dependency(const Index_t thread, const Index_t register_idx,
                    const RuntimeValue &value,
                    OnActiveBlock &&on_activate_block) {
  registers[thread][register_idx] = value;
  cl::sycl::atomic<int> atomic_dep_count(
      (cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(
          num_outstanding_dependencies)));
  if (atomic_dep_count.fetch_add(-1) == 1) {
    on_activate_block(m_handle);
  }
}
} // namespace FunGPU::EvaluatorV2
