#pragma once

#include "Core/EvaluatorV2/Instruction.h"
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/PortableMemPool.hpp"
#include "Core/Types.hpp"
#include "Core/Visitor.hpp"
#include <cstdint>

namespace FunGPU::EvaluatorV2 {
template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
class RuntimeBlock : public PortableMemPool::EnableHandleFromThis<
                         RuntimeBlock<RegistersPerThread, ThreadsPerBlock>> {
public:
  static constexpr Index_t NumRegistersPerThread = RegistersPerThread;
  static constexpr Index_t NumThreadsPerBlock = ThreadsPerBlock;

  struct TargetAddress {
    PortableMemPool::Handle<RuntimeBlock> block;
    Index_t thread;
    Index_t register_idx;
  };

  struct BlockMetadata {
    BlockMetadata(const PortableMemPool::Handle<RuntimeBlock> block,
                  const PortableMemPool::ArrayHandle<Instruction> instructions,
                  const Index_t num_threads)
        : block(block), instructions(instructions), num_threads(num_threads) {}
    BlockMetadata() = default;

    PortableMemPool::Handle<RuntimeBlock> block;
    PortableMemPool::ArrayHandle<Instruction> instructions;
    Index_t num_threads;
  };

  struct BlockExecGroup {
    BlockExecGroup(
        const PortableMemPool::ArrayHandle<BlockMetadata> &block_descs,
        const Index_t max_num_instructions)
        : block_descs(block_descs), max_num_instructions(max_num_instructions) {
    }
    BlockExecGroup() = default;

    PortableMemPool::ArrayHandle<BlockMetadata> block_descs;
    Index_t max_num_instructions = 0;
  };

  enum class Status { READY, STALLED, COMPLETE };

  using InstructionLocalMemAccessor =
      cl::sycl::accessor<Instruction, 2, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;

  explicit RuntimeBlock(
      const PortableMemPool::ArrayHandle<Instruction> instructions,
      const Index_t num_threads)
      : instruction_ref(instructions), num_threads(num_threads) {}

  template <typename OnIndirectCall, typename OnActivateBlock>
  Status evaluate(const Index_t block_idx, const Index_t thread,
                  PortableMemPool::DeviceAccessor_t mem_pool,
                  const InstructionLocalMemAccessor instructions,
                  const Index_t instruction_count,
                  OnIndirectCall &&on_indirect_call,
                  OnActivateBlock &&on_activate_block);
  RuntimeValue result(Index_t block_idx, Index_t thread,
                      InstructionLocalMemAccessor,
                      Index_t num_instructions) const;

  Index_t last_write_location(Index_t block_idx, Index_t thread,
                              InstructionLocalMemAccessor,
                              Index_t num_instructions) const;

  template <typename OnActivateBlock>
  void fill_dependency(const Index_t thread, const Index_t register_idx,
                       const RuntimeValue value,
                       OnActivateBlock &&on_activate_block);

  BlockMetadata block_metadata() const {
    return BlockMetadata(m_handle, instruction_ref, num_threads);
  }

  std::array<std::array<RuntimeValue, RegistersPerThread>, ThreadsPerBlock>
      registers;
  PortableMemPool::Handle<RuntimeBlock> m_handle;
  PortableMemPool::ArrayHandle<Instruction> instruction_ref;
  TargetAddress target_data[ThreadsPerBlock];
  Index_t num_threads;
  Index_t cur_cycle = 0;
  int num_outstanding_dependencies = 0;
};

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
Index_t RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::last_write_location(
    const Index_t block_idx, const Index_t thread,
    const InstructionLocalMemAccessor all_instructions,
    const Index_t instruction_count) const {
  const auto extract_target_register = Visitor{[]<typename DerivedInstruction>(
      const DerivedInstruction
          &instr) requires HasTargetRegister<DerivedInstruction>{
      return instr.target_register;
}
, [](const auto &) {
  // TODO error, should not happen
  return Index_t(-1);
}
}; // namespace FunGPU::EvaluatorV2

const auto &register_set = registers[thread];
const auto &instructions = all_instructions[block_idx];
if (instruction_count < 3 ||
    instructions[instruction_count - 3].type != InstructionType::IF) {
  const auto prev_instruction = instructions[instruction_count - 1];
  return visit(prev_instruction, extract_target_register, [](const auto &) {});
}
const auto &if_instr = instructions[instruction_count - 3].data.if_val;
const auto is_pred_true =
    static_cast<bool>(register_set[if_instr.predicate].data.float_val);
const auto last_instr = is_pred_true ? instructions[if_instr.goto_true]
                                     : instructions[if_instr.goto_false];
return visit(last_instr, extract_target_register, [](const auto &) {});
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
RuntimeValue RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::result(
    const Index_t block_idx, const Index_t thread,
    const InstructionLocalMemAccessor instructions,
    const Index_t num_instructions) const {
  const auto &register_set = registers[thread];
  return register_set[last_write_location(block_idx, thread, instructions,
                                          num_instructions)];
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
template <typename OnIndirectCall, typename OnActivateBlock>
auto RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::evaluate(
    const Index_t block_idx, const Index_t thread,
    PortableMemPool::DeviceAccessor_t mem_pool,
    const InstructionLocalMemAccessor all_instructions,
    const Index_t num_instructions, OnIndirectCall &&on_indirect_call,
    OnActivateBlock &&on_activate_block) -> Status {
  auto &register_set = registers[thread];
  const auto &instructions = all_instructions[block_idx];
#define HANDLE_BINARY_OP(TYPE, OP)                                             \
  [&](const TYPE &type) {                                                      \
    auto &target_register = register_set[type.target_register];                \
    target_register.type = RuntimeValue::Type::FLOAT;                          \
    target_register.data.float_val =                                           \
        register_set[type.lhs]                                                 \
            .data.float_val OP register_set[type.rhs]                          \
            .data.float_val;                                                   \
    return Status::READY;                                                      \
  }

  const auto allocate_arg_values = [&](const auto& call_indirect) {
    const auto arg_indices_handle = call_indirect.arg_indices.unpack();
    const auto *arg_indices = mem_pool[0].derefHandle(arg_indices_handle);
    auto arg_values =
        mem_pool[0].AllocArray<RuntimeValue>(arg_indices_handle.GetCount());
    auto *arg_values_data = mem_pool[0].derefHandle(arg_values);
    for (Index_t i = 0; i < arg_indices_handle.GetCount(); ++i) {
      arg_values_data[i] = register_set[arg_indices[i]];
    }
    return arg_values;
  };

  const auto non_control_flow_handlers = Visitor{
      HANDLE_BINARY_OP(Add, +),
      HANDLE_BINARY_OP(Sub, -),
      HANDLE_BINARY_OP(Mul, *),
      HANDLE_BINARY_OP(Div, /),
      HANDLE_BINARY_OP(Equal, ==),
      HANDLE_BINARY_OP(GreaterThan, >),
      [&](const Floor &floor) {
        auto &target_register = register_set[floor.target_register];
        target_register.type = RuntimeValue::Type::FLOAT;
        target_register.data.float_val =
            cl::sycl::floor(register_set[floor.arg].data.float_val);
        return Status::READY;
      },
      [&](const Remainder &remainder) {
        auto &target_register = register_set[remainder.target_register];
        target_register.type = RuntimeValue::Type::FLOAT;
        target_register.data.float_val =
            cl::sycl::fmod(register_set[remainder.lhs].data.float_val,
                           register_set[remainder.rhs].data.float_val);
        return Status::READY;
      },
      [&](const Expt &expr) {
        auto &target_register = register_set[expr.target_register];
        target_register.type = RuntimeValue::Type::FLOAT;
        target_register.data.float_val =
            cl::sycl::pow(register_set[expr.lhs].data.float_val,
                          register_set[expr.rhs].data.float_val);
        return Status::READY;
      },
      [&](const AssignConstant &assign_constant) {
        auto &target_register = register_set[assign_constant.target_register];
        target_register.type = RuntimeValue::Type::FLOAT;
        target_register.data.float_val = assign_constant.constant;
        return Status::READY;
      },
      [&](const Assign &assign) {
        register_set[assign.target_register] =
            register_set[assign.source_register];
        return Status::READY;
      },
      [&](const CreateLambda &create_lambda) {
        const auto captured_indices_handle =
            create_lambda.captured_indices.unpack();
        auto *captured_indices =
            mem_pool[0].derefHandle(captured_indices_handle);
        auto &target_register = register_set[create_lambda.target_register];
        auto captured_values = mem_pool[0].AllocArray<RuntimeValue>(
            captured_indices_handle.GetCount());
        auto *captured_values_data = mem_pool[0].derefHandle(captured_values);
        target_register.type = RuntimeValue::Type::LAMBDA;
        target_register.data.function_val =
            FunctionValue(create_lambda.block_idx, captured_values);
        for (Index_t i = 0; i < captured_indices_handle.GetCount(); ++i) {
          captured_values_data[i] = register_set[captured_indices[i]];
        }
        return Status::READY;
      },
      [&](const CallIndirect &call_indirect) {
        const auto arg_values = allocate_arg_values(call_indirect);
        cl::sycl::atomic<int, cl::sycl::access::address_space::local_space>
        atomic_dep_count(
            (cl::sycl::multi_ptr<int,
                                 cl::sycl::access::address_space::local_space>(
                &num_outstanding_dependencies)));
        atomic_dep_count.fetch_add(1);
        const auto function_val =
            register_set[call_indirect.lambda_idx].data.function_val;
        on_indirect_call(m_handle, function_val, thread,
                        call_indirect.target_register, arg_values);
        return Status::READY;
      },
      [&](const BlockingCallIndirect &blocking_call_indirect) {
        const auto arg_values = allocate_arg_values(blocking_call_indirect);
        const auto function_val =
            register_set[blocking_call_indirect.lambda_idx].data.function_val;
        const auto &target = target_data[thread];
        on_indirect_call(target.block, function_val, target.thread,
                        target.register_idx, arg_values);
        return Status::COMPLETE;
      },
      [&](const InstructionBarrier &) { return Status::STALLED; }};

#undef HANDLE_BINARY_OP

  auto status = Status::READY;
  for (; status == Status::READY; ++cur_cycle) {
    if (cur_cycle > num_instructions) {
      status = Status::COMPLETE;
      continue;
    } else if (cur_cycle == num_instructions ||
               (num_instructions >= 3 &&
                instructions[num_instructions - 3].type ==
                    InstructionType::IF &&
                cur_cycle > num_instructions - 3)) {
      const auto &target = target_data[thread];
      if (target.block == PortableMemPool::Handle<RuntimeBlock>()) {
        status = Status::COMPLETE;
        continue;
      }
      auto &target_block = *mem_pool[0].derefHandle(target.block);
      const auto last_write_loc_for_fill = last_write_location(
          block_idx, thread, all_instructions, num_instructions);
      target_block.fill_dependency(target.thread, target.register_idx,
                                   register_set[last_write_loc_for_fill],
                                   on_activate_block);
      status = Status::COMPLETE;
      continue;
    }

    const auto &instruction = instructions[cur_cycle];
    status = visit(
        instruction,
        [&](const auto &instr) {
          if constexpr (std::is_same_v<If,
                                       std::remove_cvref_t<decltype(instr)>>) {
            const auto &next_instr =
                static_cast<bool>(register_set[instr.predicate].data.float_val)
                    ? instructions[instr.goto_true]
                    : instructions[instr.goto_false];
            return visit(
                next_instr,
                [&](const auto &derived_next_instr) {
                  if constexpr (std::is_same_v<If, std::remove_cvref_t<decltype(
                                                       derived_next_instr)>>) {
                    // TODO error, this should never happen
                    return Status::READY;
                  } else {
                    return non_control_flow_handlers(derived_next_instr);
                  }
                },
                [](const auto &) {});
          } else {
            return non_control_flow_handlers(instr);
          }
        },
        [](const auto &) {});
  }
  return status;
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
template <typename OnActivateBlock>
void RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::fill_dependency(
    const Index_t thread, const Index_t register_idx, const RuntimeValue value,
    OnActivateBlock &&on_activate_block) {
  registers[thread][register_idx] = value;
  cl::sycl::atomic<int> atomic_dep_count(
      (cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(
          &num_outstanding_dependencies)));
  const auto prev_count = atomic_dep_count.fetch_sub(1);
  if (prev_count == 1) {
    on_activate_block(m_handle);
  }
}
} // namespace FunGPU::EvaluatorV2
