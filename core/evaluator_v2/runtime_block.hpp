#pragma once

#include "core/concepts.hpp"
#include "core/evaluator_v2/instruction.hpp"
#include "core/evaluator_v2/program.hpp"
#include "core/evaluator_v2/runtime_value.hpp"
#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"
#include "core/visitor.hpp"
#include <cstdint>
#include <optional>

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

  enum class Status : std::uint8_t { READY, STALLED, COMPLETE };

  using InstructionLocalMemAccessor =
      cl::sycl::accessor<Instruction, 2, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;
  using PreAllocatedRuntimeValuesPerThread = std::array<
      PortableMemPool::ArrayHandle<PortableMemPool::ArrayHandle<RuntimeValue>>,
      ThreadsPerBlock>;

  template <cl::sycl::access::target ACCESS_TARGET>
  static std::optional<PreAllocatedRuntimeValuesPerThread>
  pre_allocate_runtime_values(
      Index_t num_threads,
      cl::sycl::accessor<PortableMemPool, 1, cl::sycl::access::mode::read_write,
                         ACCESS_TARGET>,
      Program, Index_t lambda_idx);
  template <cl::sycl::access::target ACCESS_TARGET>
  static void deallocate_runtime_values_for_thread(
      cl::sycl::accessor<PortableMemPool, 1, cl::sycl::access::mode::read_write,
                         ACCESS_TARGET>,
      const PreAllocatedRuntimeValuesPerThread &pre_allocated_runtime_values,
      Index_t thread_idx);

  explicit RuntimeBlock(
      const PortableMemPool::ArrayHandle<Instruction> instructions,
      const PreAllocatedRuntimeValuesPerThread pre_allocated_runtime_values,
      const Index_t num_threads)
      : instruction_ref(instructions),
        pre_allocated_runtime_values(pre_allocated_runtime_values),
        num_threads(num_threads) {}

  // RuntimeBlock must be loaded into local memory before invoking evaluate
  template <typename OnIndirectCall, typename OnActivateBlock>
  Status evaluate(const Index_t block_idx, const Index_t thread,
                  const cl::sycl::nd_item<1> &,
                  PortableMemPool::DeviceAccessor_t mem_pool,
                  const InstructionLocalMemAccessor instructions,
                  const Index_t instruction_count, Index_t num_threads,
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

  void deallocate_runtime_values_array_for_thread(
      PortableMemPool::DeviceAccessor_t mem_pool, Index_t thread);

  BlockMetadata block_metadata() const {
    return BlockMetadata(m_handle, instruction_ref, num_threads);
  }

  std::array<std::array<RuntimeValue, RegistersPerThread>, ThreadsPerBlock>
      registers;
  PortableMemPool::Handle<RuntimeBlock> m_handle;
  PortableMemPool::ArrayHandle<Instruction> instruction_ref;
  PreAllocatedRuntimeValuesPerThread pre_allocated_runtime_values;
  TargetAddress target_data[ThreadsPerBlock];
  Index_t num_threads;
  Index_t cur_cycle = 0;
  Index_t pre_allocated_runtime_values_idx = 0;
  int num_outstanding_dependencies = 0;
};

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
void RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::
    deallocate_runtime_values_array_for_thread(
        PortableMemPool::DeviceAccessor_t mem_pool, const Index_t thread_idx) {
  if (const auto &prealloced_values = pre_allocated_runtime_values[thread_idx];
      prealloced_values.get_count() != 0) {
    mem_pool[0].dealloc_array(prealloced_values);
  }
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
Index_t RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::last_write_location(
    const Index_t block_idx, const Index_t thread,
    const InstructionLocalMemAccessor all_instructions,
    const Index_t instruction_count) const {
  const auto extract_target_register =
      Visitor{[]<typename DerivedInstruction>(const DerivedInstruction &instr)
                requires HasTargetRegister<DerivedInstruction>
              { return instr.target_register; },
              [](const auto &) {
                // TODO error, should not happen
                return Index_t(-1);
              }}; // namespace FunGPU::EvaluatorV2

  const auto &register_set = registers[thread];
  const auto &instructions = all_instructions[block_idx];
  if (instruction_count < 3 ||
      instructions[instruction_count - 3].type != InstructionType::IF) {
    const auto prev_instruction = instructions[instruction_count - 1];
    return visit(prev_instruction, extract_target_register,
                 [](const auto &) { return Index_t(-1); });
  }
  const auto &if_instr = instructions[instruction_count - 3].data.if_val;
  const auto is_pred_true =
      static_cast<bool>(register_set[if_instr.predicate].data.float_val);
  const auto last_instr = is_pred_true ? instructions[if_instr.goto_true]
                                       : instructions[if_instr.goto_false];
  return visit(last_instr, extract_target_register,
               [](const auto &) { return Index_t(-1); });
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
    const cl::sycl::nd_item<1> &itm, PortableMemPool::DeviceAccessor_t mem_pool,
    const InstructionLocalMemAccessor all_instructions,
    const Index_t num_instructions, const Index_t num_threads,
    OnIndirectCall &&on_indirect_call,
    OnActivateBlock &&on_activate_block) -> Status {
  auto local_cycle = cur_cycle;
  const auto pre_allocated_rvs = pre_allocated_runtime_values[thread];
  auto local_pre_allocated_runtime_values_idx =
      pre_allocated_runtime_values_idx;
  auto status = Status::COMPLETE;
  if (thread < num_threads) {
    auto &register_set = registers[thread];
    const auto &instructions = all_instructions[block_idx];
#define HANDLE_BINARY_OP(TYPE, OP)                                             \
  [&](const TYPE &type) {                                                      \
    auto &target_register = register_set[type.target_register];                \
    target_register.data.float_val =                                           \
        register_set[type.lhs]                                                 \
            .data.float_val OP register_set[type.rhs]                          \
            .data.float_val;                                                   \
    return Status::READY;                                                      \
  }

    const auto allocate_arg_values = [&](const auto &call_indirect) {
      const auto arg_indices_handle = call_indirect.arg_indices.unpack();
      if (arg_indices_handle.get_count() == 0) {
        ++local_pre_allocated_runtime_values_idx;
        return PortableMemPool::ArrayHandle<RuntimeValue>();
      }
      const auto *arg_indices = mem_pool[0].deref_handle(arg_indices_handle);
      auto *pre_allocated_runtime_values_data =
          mem_pool[0].deref_handle(pre_allocated_rvs);
      auto arg_values = pre_allocated_runtime_values_data
          [local_pre_allocated_runtime_values_idx++];
      auto *arg_values_data = mem_pool[0].deref_handle(arg_values);
      for (Index_t i = 0; i < arg_indices_handle.get_count(); ++i) {
        arg_values_data[i] = register_set[arg_indices[i]];
      }
      return arg_values;
    };

    const auto deallocate_runtime_values = Visitor{
        [&](const OneOf<CallIndirect, BlockingCallIndirect, CreateLambda> auto
                &op_with_rvs) {
          auto *pre_allocated_runtime_values_data =
              mem_pool[0].deref_handle(pre_allocated_rvs);
          if (const auto &pre_allocated_runtime_values =
                  pre_allocated_runtime_values_data
                      [local_pre_allocated_runtime_values_idx];
              pre_allocated_runtime_values.get_count() != 0) {
            mem_pool[0].dealloc_array(pre_allocated_runtime_values);
          }
          return true;
        },
        [](const OneOf<Add, Sub, Mul, Div, Equal, GreaterThan, Remainder, Expt,
                       If, Assign, AssignConstant, Floor,
                       InstructionBarrier> auto &) { return false; }};

    const auto non_control_flow_handlers = Visitor{
        HANDLE_BINARY_OP(Add, +),
        HANDLE_BINARY_OP(Sub, -),
        HANDLE_BINARY_OP(Mul, *),
        HANDLE_BINARY_OP(Div, /),
        HANDLE_BINARY_OP(Equal, ==),
        HANDLE_BINARY_OP(GreaterThan, >),
        [&](const Floor &floor) {
          auto &target_register = register_set[floor.target_register];
          target_register.data.float_val =
              cl::sycl::floor(register_set[floor.arg].data.float_val);
          return Status::READY;
        },
        [&](const Remainder &remainder) {
          auto &target_register = register_set[remainder.target_register];
          target_register.data.float_val =
              cl::sycl::fmod(register_set[remainder.lhs].data.float_val,
                             register_set[remainder.rhs].data.float_val);
          return Status::READY;
        },
        [&](const Expt &expr) {
          auto &target_register = register_set[expr.target_register];
          target_register.data.float_val =
              cl::sycl::pow(register_set[expr.lhs].data.float_val,
                            register_set[expr.rhs].data.float_val);
          return Status::READY;
        },
        [&](const AssignConstant &assign_constant) {
          auto &target_register = register_set[assign_constant.target_register];
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
          auto &target_register = register_set[create_lambda.target_register];
          auto *rv_data = mem_pool[0].deref_handle(pre_allocated_rvs);
          const auto captured_values =
              rv_data[local_pre_allocated_runtime_values_idx++];
          target_register.data.function_val =
              FunctionValue(create_lambda.block_idx, captured_values);
          if (captured_indices_handle.get_count() != 0) {
            auto *captured_values_data =
                mem_pool[0].deref_handle(captured_values);
            auto *captured_indices =
                mem_pool[0].deref_handle(captured_indices_handle);
            for (Index_t i = 0; i < captured_indices_handle.get_count(); ++i) {
              captured_values_data[i] = register_set[captured_indices[i]];
            }
          }
          return Status::READY;
        },
        [&](const CallIndirect &call_indirect) {
          const auto arg_values = allocate_arg_values(call_indirect);
          cl::sycl::atomic_ref<int, cl::sycl::memory_order::seq_cst,
                               cl::sycl::memory_scope::work_group,
                               cl::sycl::access::address_space::local_space>
              atomic_dep_count(num_outstanding_dependencies);
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

    status = Status::READY;
    while (status == Status::READY) {
      if (local_cycle > num_instructions) {
        status = Status::COMPLETE;
        continue;
      } else if (local_cycle == num_instructions ||
                 (num_instructions >= 3 &&
                  instructions[num_instructions - 3].type ==
                      InstructionType::IF &&
                  local_cycle > num_instructions - 3)) {
        const auto &target = target_data[thread];
        if (target.block == PortableMemPool::Handle<RuntimeBlock>()) {
          status = Status::COMPLETE;
          continue;
        }
        auto &target_block = *mem_pool[0].deref_handle(target.block);
        const auto last_write_loc_for_fill = last_write_location(
            block_idx, thread, all_instructions, num_instructions);
        target_block.fill_dependency(target.thread, target.register_idx,
                                     register_set[last_write_loc_for_fill],
                                     on_activate_block);
        status = Status::COMPLETE;
        continue;
      }

      const auto &instruction = instructions[local_cycle];
      status = visit(
          instruction,
          [&](const auto &instr) {
            if constexpr (std::is_same_v<
                              If, std::remove_cvref_t<decltype(instr)>>) {
              const auto is_branch_true = static_cast<bool>(
                  register_set[instr.predicate].data.float_val);
              const auto &next_instr = is_branch_true
                                           ? instructions[instr.goto_true]
                                           : instructions[instr.goto_false];
              if (!is_branch_true) {
                if (visit(instructions[instr.goto_true],
                          deallocate_runtime_values,
                          [](const auto &) { return false; })) {
                  ++local_pre_allocated_runtime_values_idx;
                }
              }
              const auto result = visit(
                  next_instr,
                  [&](const auto &derived_next_instr) {
                    if constexpr (std::is_same_v<
                                      If, std::remove_cvref_t<
                                              decltype(derived_next_instr)>>) {
                      // TODO error, this should never happen
                      return Status::READY;
                    } else {
                      return non_control_flow_handlers(derived_next_instr);
                    }
                  },
                  [](const auto &) { return Status::READY; });
              if (is_branch_true) {
                if (visit(instructions[instr.goto_false],
                          deallocate_runtime_values,
                          [](const auto &) { return false; })) {
                  ++local_pre_allocated_runtime_values_idx;
                }
              }
              return result;
            } else {
              return non_control_flow_handlers(instr);
            }
          },
          [](const auto &) { return Status::READY; });
      ++local_cycle;
    }
  }
  cl::sycl::group_barrier(itm.get_group());
  if (thread == 0) {
    cur_cycle = local_cycle;
    pre_allocated_runtime_values_idx = local_pre_allocated_runtime_values_idx;
  }
  return status;
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
template <typename OnActivateBlock>
void RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::fill_dependency(
    const Index_t thread, const Index_t register_idx, const RuntimeValue value,
    OnActivateBlock &&on_activate_block) {
  registers[thread][register_idx] = value;
  cl::sycl::atomic_ref<int, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      atomic_dep_count(num_outstanding_dependencies);
  const auto prev_count = atomic_dep_count.fetch_sub(1);
  if (prev_count == 1) {
    on_activate_block(m_handle);
  }
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
template <cl::sycl::access::target ACCESS_TARGET>
void RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::
    deallocate_runtime_values_for_thread(
        cl::sycl::accessor<PortableMemPool, 1,
                           cl::sycl::access::mode::read_write, ACCESS_TARGET>
            mem_pool_acc,
        const PreAllocatedRuntimeValuesPerThread &pre_allocated_runtime_values,
        Index_t thread_idx) {
  if (pre_allocated_runtime_values[thread_idx].get_count() == 0) {
    return;
  }
  const auto *pre_allocated_rvs_data =
      mem_pool_acc[0].deref_handle(pre_allocated_runtime_values[thread_idx]);
  for (Index_t j = 0; j < pre_allocated_runtime_values[thread_idx].get_count();
       ++j) {
    if (pre_allocated_rvs_data[j].get_count() != 0) {
      mem_pool_acc[0].dealloc_array(pre_allocated_rvs_data[j]);
    }
  }
  mem_pool_acc[0].dealloc_array(pre_allocated_runtime_values[thread_idx]);
}

template <Index_t RegistersPerThread, Index_t ThreadsPerBlock>
template <cl::sycl::access::target ACCESS_TARGET>
auto RuntimeBlock<RegistersPerThread, ThreadsPerBlock>::
    pre_allocate_runtime_values(
        const Index_t num_threads,
        cl::sycl::accessor<PortableMemPool, 1,
                           cl::sycl::access::mode::read_write, ACCESS_TARGET>
            mem_pool_acc,
        const Program program, const Index_t lambda_idx)
        -> std::optional<PreAllocatedRuntimeValuesPerThread> {
  const auto &instructions_data =
      mem_pool_acc[0].deref_handle(program)[lambda_idx];

  PreAllocatedRuntimeValuesPerThread pre_allocated_rvs;

  const auto deallocate_runtime_values_up_to_idx =
      [&](const Index_t thread_idx) {
        for (Index_t i = 0; i < thread_idx; ++i) {
          deallocate_runtime_values_for_thread<ACCESS_TARGET>(
              mem_pool_acc, pre_allocated_rvs, i);
        }
      };

  for (Index_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    const auto pre_allocated_rvs_handle =
        mem_pool_acc[0]
            .template alloc_array<PortableMemPool::ArrayHandle<RuntimeValue>>(
                instructions_data.instruction_properties
                    .num_runtime_values_per_op.get_count());
    if (pre_allocated_rvs_handle ==
            PortableMemPool::ArrayHandle<
                PortableMemPool::ArrayHandle<RuntimeValue>>() &&
        instructions_data.instruction_properties.num_runtime_values_per_op
                .get_count() != 0) {
      deallocate_runtime_values_up_to_idx(thread_idx);
      return std::nullopt;
    }
    pre_allocated_rvs[thread_idx] = pre_allocated_rvs_handle;
    const auto *num_rvs_per_op = mem_pool_acc[0].deref_handle(
        instructions_data.instruction_properties.num_runtime_values_per_op);
    auto *pre_allocated_rv_data =
        mem_pool_acc[0].deref_handle(pre_allocated_rvs_handle);
    for (Index_t op_idx = 0; op_idx < pre_allocated_rvs_handle.get_count();
         ++op_idx) {
      const auto pre_allocated_values_for_op =
          mem_pool_acc[0].template alloc_array<RuntimeValue>(
              num_rvs_per_op[op_idx]);
      if (pre_allocated_values_for_op ==
              PortableMemPool::ArrayHandle<RuntimeValue>() &&
          num_rvs_per_op[op_idx] != 0) {
        deallocate_runtime_values_up_to_idx(thread_idx + 1);
        return std::nullopt;
      }
      pre_allocated_rv_data[op_idx] = pre_allocated_values_for_op;
    }
  }
  return pre_allocated_rvs;
}
} // namespace FunGPU::EvaluatorV2
