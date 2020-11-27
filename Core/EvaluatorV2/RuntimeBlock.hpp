#pragma once

#include "Core/EvaluatorV2/Instruction.h"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/Types.hpp"
#include "Core/Visitor.hpp"
#include "Core/PortableMemPool.hpp"
#include <cstdint>

namespace FunGPU::EvaluatorV2 {
template <Index_t RegistersPerThread,
          Index_t ThreadsPerBlock>
class RuntimeBlock
    : public PortableMemPool::EnableHandleFromThis<RuntimeBlock<RegistersPerThread, ThreadsPerBlock>> {
public:
  struct TargetAddress {
    PortableMemPool::Handle<RuntimeBlock> block;
    Index_t thread;
    Index_t register_idx;
  };

  struct BlockMetadata {
    BlockMetadata(const PortableMemPool::Handle<RuntimeBlock> block,
      const PortableMemPool::ArrayHandle<Instruction> instructions) : block(block),
        instructions(instructions) {}

    const PortableMemPool::Handle<RuntimeBlock> block;
    const PortableMemPool::ArrayHandle<Instruction> instructions;
  };

  using InstructionLocalMemAccessor = cl::sycl::accessor<Instruction, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

  template <typename OnIndirectCall, typename OnActivateBlock> bool step(
          const Index_t thread, PortableMemPool::DeviceAccessor_t mem_pool,
          const Index_t cur_cycle,
         const InstructionLocalMemAccessor instructions,
         OnIndirectCall &&on_indirect_call,
         OnActivateBlock &&on_activate_block);
  RuntimeValue result(Index_t thread, InstructionLocalMemAccessor) const;

  Index_t last_write_location(Index_t thread, InstructionLocalMemAccessor) const;

  template<typename OnActivateBlock>
  void fill_dependency(const Index_t thread, const Index_t register_idx,
                    const RuntimeValue value,
                    OnActivateBlock &&on_activate_block); 

  RuntimeValue registers[ThreadsPerBlock][RegistersPerThread];
  PortableMemPool::Handle<RuntimeBlock> m_handle;
  TargetAddress target_data[ThreadsPerBlock];
  int num_outstanding_dependencies = 0;
};

template<Index_t RegistersPerThread,
  Index_t ThreadsPerBlock>
Index_t RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::last_write_location(const Index_t thread,
  const InstructionLocalMemAccessor instructions) const {
  const auto extract_target_register = [](const auto& instr) {
    if constexpr (std::is_same_v<If, std::remove_cvref_t<decltype(instr)>>) {
      // TODO error, should not happen
      return Index_t(-1);
    } else {
      return instr.target_register;
    }
  };

  const auto &register_set = registers[thread];
  if (instructions.get_count() < 3 || instructions[instructions.get_count() - 3].type != InstructionType::IF) {
    const auto prev_instruction = instructions[instructions.get_count() - 1];
    return visit(prev_instruction, extract_target_register, [](const auto&){});
  }
  const auto& if_instr = instructions[instructions.get_count() - 3].data.if_val;
  const auto last_instr = static_cast<bool>(register_set[if_instr.predicate].data.float_val) ? 
    instructions[if_instr.goto_true] : instructions[if_instr.goto_false];
  return visit(last_instr, extract_target_register, [](const auto&){});
}

template<Index_t RegistersPerThread,
  Index_t ThreadsPerBlock>
RuntimeValue RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::result(const Index_t thread,
   const InstructionLocalMemAccessor instructions) const {
  const auto &register_set = registers[thread];
  return register_set[last_write_location(thread, instructions)];
}

template <Index_t RegistersPerThread,
          Index_t ThreadsPerBlock>
template <typename OnIndirectCall, typename OnActivateBlock>
bool RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::
    step(const Index_t thread, PortableMemPool::DeviceAccessor_t mem_pool,
         const Index_t cur_cycle,
         const InstructionLocalMemAccessor instructions,
         OnIndirectCall &&on_indirect_call,
         OnActivateBlock &&on_activate_block) {
  const auto num_instructions = instructions.get_count();
  auto &register_set = registers[thread];

  if (cur_cycle > num_instructions) {
    return false;
  } else if (cur_cycle == num_instructions || 
    (instructions[instructions.get_count() - 3].type == InstructionType::IF && cur_cycle > instructions.get_count() - 3)) {
    const auto &target = target_data[thread];
    if (target.block == PortableMemPool::Handle<RuntimeBlock>()) {
      return false;
    }
    auto &target_block = *mem_pool[0].derefHandle(target.block);
    target_block.fill_dependency(target.thread, target.register_idx,
                                 register_set[last_write_location(thread, instructions)],
                                 on_activate_block);
    return false;
  }

  const auto &instruction = instructions[cur_cycle];

#define HANDLE_BINARY_OP(TYPE, OP)                                             \
  [&](const TYPE &type) {                                                      \
    auto &target_register = register_set[type.target_register];                         \
    target_register.type = RuntimeValue::Type::FLOAT;                          \
    target_register.data.float_val =                                                    \
        register_set[type.lhs].data.float_val OP register_set[type.rhs].data.float_val; \
  }

  const auto non_control_flow_handlers = Visitor{HANDLE_BINARY_OP(Add, +), HANDLE_BINARY_OP(Sub, -),
                HANDLE_BINARY_OP(Mul, *), HANDLE_BINARY_OP(Div, /),
                HANDLE_BINARY_OP(Equal, ==), HANDLE_BINARY_OP(GreaterThan, >),
                [&](const Floor& floor) {
                  auto& target_register = register_set[floor.target_register];
                  target_register.type = RuntimeValue::Type::FLOAT;
                  target_register.data.float_val = cl::sycl::floor(register_set[floor.arg].data.float_val);
                },
                [&](const Remainder& remainder) {
                  auto& target_register = register_set[remainder.target_register];
                  target_register.type = RuntimeValue::Type::FLOAT;
                  target_register.data.float_val = cl::sycl::fmod(register_set[remainder.lhs].data.float_val, 
                    register_set[remainder.rhs].data.float_val);
                },
                [&](const Expt& expr) {
                  auto& target_register = register_set[expr.target_register];
                  target_register.type = RuntimeValue::Type::FLOAT;
                  target_register.data.float_val = cl::sycl::pow(register_set[expr.lhs].data.float_val,
                    register_set[expr.rhs].data.float_val);
                },
                [&](const AssignConstant &assign_constant) {
                  auto &target_register =
                      register_set[assign_constant.target_register];
                  target_register.type = RuntimeValue::Type::FLOAT;
                  target_register.data.float_val = assign_constant.constant;
                },
                [&](const Assign& assign) {
                  register_set[assign.target_register] = register_set[assign.source_register];
                },
                [&](const CreateLambda &create_lambda) {
                  const auto captured_indices_handle = create_lambda.captured_indices.unpack();
                  auto *captured_indices =
                      mem_pool[0].derefHandle(captured_indices_handle);
                  auto &target_register =
                      register_set[create_lambda.target_register];
                  auto captured_values = mem_pool[0].AllocArray<RuntimeValue>(
                      captured_indices_handle.GetCount());
                  auto *captured_values_data =
                      mem_pool[0].derefHandle(captured_values);
                  for (Index_t i = 0;
                       i < captured_indices_handle.GetCount(); ++i) {
                    captured_values_data[i] = register_set[captured_indices[i]];
                  }
                  target_register.type = RuntimeValue::Type::LAMBDA;
                  target_register.data.function_val =
                      FunctionValue(create_lambda.block_idx, captured_values);
                },
                [&](const CallIndirect &call_indirect) {
                  const auto arg_indices_handle = call_indirect.arg_indices.unpack();
                  const auto *arg_indices =
                      mem_pool[0].derefHandle(arg_indices_handle);
                  auto arg_values = mem_pool[0].AllocArray<RuntimeValue>(
                      arg_indices_handle.GetCount());
                  auto *arg_values_data = mem_pool[0].derefHandle(arg_values);
                  for (Index_t i = 0; i < arg_indices_handle.GetCount();
                       ++i) {
                    arg_values_data[i] = register_set[arg_indices[i]];
                  }
                  cl::sycl::atomic<int, cl::sycl::access::address_space::local_space> atomic_dep_count(
                      (cl::sycl::multi_ptr<
                          int, cl::sycl::access::address_space::local_space>(
                          &num_outstanding_dependencies)));
                  atomic_dep_count.fetch_add(1);
                  const auto function_val = register_set[call_indirect.lambda_idx].data.function_val;
                  on_indirect_call(m_handle, function_val,
                                   thread,  call_indirect.target_register, arg_values);
                }};

  visit(instruction, [&](const auto& instr) {
    if constexpr (std::is_same_v<If, std::remove_cvref_t<decltype(instr)>>) {
      const auto& next_instr = static_cast<bool>(register_set[instr.predicate].data.float_val) ? instructions[instr.goto_true] : instructions[instr.goto_false];
      visit(next_instr, [&](const auto& next_instr) {
        if constexpr (std::is_same_v<If, std::remove_cvref_t<decltype(next_instr)>>) {
          // TODO error, this should never happen
        } else {
          non_control_flow_handlers(next_instr);
        }
      }, [](const auto&){});
    } else {
      non_control_flow_handlers(instr);
    }
  }, [](const auto&) {});
#undef HANDLE_BINARY_OP

  return true;
}

template <Index_t RegistersPerThread,
          Index_t ThreadsPerBlock>
template <typename OnActivateBlock>
void RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::
    fill_dependency(const Index_t thread, const Index_t register_idx,
                    const RuntimeValue value,
                    OnActivateBlock &&on_activate_block) {
  registers[thread][register_idx] = value;
  cl::sycl::atomic<int> atomic_dep_count(
      (cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(
          &num_outstanding_dependencies)));
  if (atomic_dep_count.fetch_sub(1) == 1) {
    on_activate_block(m_handle);
  }
}
} // namespace FunGPU::EvaluatorV2
