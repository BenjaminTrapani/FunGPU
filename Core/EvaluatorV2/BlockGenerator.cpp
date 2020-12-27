#include "Core/EvaluatorV2/BlockGenerator.h"
#include "Core/CollectAllASTNodes.hpp"
#include "Core/EvaluatorV2/Instruction.h"
#include "Core/Visitor.hpp"
#include <deque>
#include <iostream>
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {

BlockGenerator::BlockGenerator(cl::sycl::buffer<PortableMemPool> pool,
                               const Index_t registers_per_block)
    : pool_(pool), registers_per_block_(registers_per_block) {}

void BlockGenerator::assign_lambdas_block_indices(
    PortableMemPool::HostAccessor_t &mem_pool_acc,
    const Compiler::ASTNodeHandle root,
    std::map<Compiler::ASTNodeHandle, Index_t> &result, Index_t &cur_index) {
  visit(*mem_pool_acc[0].derefHandle(root),
        Visitor{[&](const Compiler::LambdaNode &bind_node) {
                  if (!result.emplace(root, cur_index++).second) {
                    throw std::invalid_argument(
                        "Multiple paths to the same lambda node in tree");
                  }
                  assign_lambdas_block_indices(
                      mem_pool_acc, bind_node.m_childExpr, result, cur_index);
                },
                [&](const auto &node) {
                  node.for_each_sub_expr(
                      mem_pool_acc, [&](const auto &sub_expr) {
                        assign_lambdas_block_indices(mem_pool_acc, sub_expr,
                                                     result, cur_index);
                      });
                }},
        [](const auto &) { throw std::invalid_argument("Unexpected node"); });
}

void BlockGenerator::compute_lambda_space_ident_to_use_count(
    const Compiler::ASTNodeHandle &node, Index_t num_bound_so_far,
    PortableMemPool::HostAccessor_t &mem_pool_acc,
    std::unordered_map<Index_t, Index_t>
        &lambda_space_ident_to_remaining_count) {
  visit(*mem_pool_acc[0].derefHandle(node),
        Visitor{
            [&](const Compiler::BindNode &bind_node) {
              for (auto idx = 0; idx < bind_node.m_bindings.GetCount(); ++idx) {
                lambda_space_ident_to_remaining_count[num_bound_so_far++] = 0;
              }
              const auto num_bound_for_bound_expr =
                  bind_node.m_type == Compiler::ASTNode::Type::BindRec
                      ? num_bound_so_far
                      : num_bound_so_far - bind_node.m_bindings.GetCount();
              const auto *binding_data =
                  mem_pool_acc[0].derefHandle(bind_node.m_bindings);
              for (auto idx = 0; idx < bind_node.m_bindings.GetCount(); ++idx) {
                compute_lambda_space_ident_to_use_count(
                    binding_data[idx], num_bound_for_bound_expr, mem_pool_acc,
                    lambda_space_ident_to_remaining_count);
              }
              compute_lambda_space_ident_to_use_count(
                  bind_node.m_childExpr, num_bound_so_far, mem_pool_acc,
                  lambda_space_ident_to_remaining_count);
            },
            [&](const Compiler::LambdaNode &lambda_node) {
              std::set<Index_t> captured_indices;
              extract_lambda_space_captured_indices(node, 0, mem_pool_acc,
                                                    captured_indices);
              for (const auto captured_index : captured_indices) {
                const auto lambda_space_index =
                    captured_index + num_bound_so_far;
                auto iter = lambda_space_ident_to_remaining_count.find(
                    lambda_space_index);
                if (iter != lambda_space_ident_to_remaining_count.end()) {
                  ++iter->second;
                } else {
                  throw std::invalid_argument(
                      "Found unbound identifier in captures of nested lambda "
                      "when computing remaining use count");
                }
              }
            },
            [&](const Compiler::IdentifierNode &ident) {
              const auto lambda_space_ident =
                  num_bound_so_far - ident.m_index - 1;
              auto iter = lambda_space_ident_to_remaining_count.find(
                  lambda_space_ident);
              if (iter != lambda_space_ident_to_remaining_count.end()) {
                ++iter->second;
              } else {
                throw std::invalid_argument("Found unbound identifier when "
                                            "computing remaining use count");
              }
            },
            [&](const auto &node) {
              node.for_each_sub_expr(mem_pool_acc, [&](const auto &elem) {
                compute_lambda_space_ident_to_use_count(
                    elem, num_bound_so_far, mem_pool_acc,
                    lambda_space_ident_to_remaining_count);
              });
            }},
        [](const auto &) { throw std::invalid_argument("Unexpected node"); });
}

void BlockGenerator::extract_lambda_space_captured_indices(
    const Compiler::ASTNodeHandle &node, Index_t num_bound_so_far,
    PortableMemPool::HostAccessor_t &mem_pool_acc,
    std::set<Index_t> &captured_indices) {
  const auto recurse_on_subexprs = [&](const auto &sub_expr) {
    sub_expr.for_each_sub_expr(mem_pool_acc, [&](const auto &elem) {
      extract_lambda_space_captured_indices(elem, num_bound_so_far,
                                            mem_pool_acc, captured_indices);
    });
  };

  visit(*mem_pool_acc[0].derefHandle(node),
        Visitor{[&](const Compiler::BindNode &bind_node) {
                  const auto is_rec =
                      bind_node.m_type == Compiler::ASTNode::Type::BindRec;
                  if (is_rec) {
                    num_bound_so_far += bind_node.m_bindings.GetCount();
                  }
                  if (bind_node.m_bindings !=
                      PortableMemPool::ArrayHandle<Compiler::ASTNodeHandle>()) {
                    const auto *binding_data =
                        mem_pool_acc[0].derefHandle(bind_node.m_bindings);
                    for (Index_t i = 0; i < bind_node.m_bindings.GetCount();
                         ++i) {
                      extract_lambda_space_captured_indices(
                          binding_data[i], num_bound_so_far, mem_pool_acc,
                          captured_indices);
                    }
                  }
                  if (!is_rec) {
                    num_bound_so_far += bind_node.m_bindings.GetCount();
                  }
                  extract_lambda_space_captured_indices(
                      bind_node.m_childExpr, num_bound_so_far, mem_pool_acc,
                      captured_indices);
                },
                [&](const Compiler::LambdaNode &lambda_node) {
                  num_bound_so_far += lambda_node.m_argCount;
                  recurse_on_subexprs(lambda_node);
                },
                [&](const Compiler::IdentifierNode &ident_node) {
                  if (ident_node.m_index >= num_bound_so_far) {
                    captured_indices.emplace(num_bound_so_far -
                                             ident_node.m_index - 1);
                  }
                },
                [&](const auto &sub_node) { recurse_on_subexprs(sub_node); }},
        [](const auto &) { throw std::invalid_argument("Unexpected node"); });
}

Lambda BlockGenerator::construct_block(
    const Compiler::ASTNodeHandle &lambda, const Compiler::ASTNodeHandle root,
    const std::map<Compiler::ASTNodeHandle, Index_t> &lambdas_to_blocks,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  std::deque<Index_t> free_indices;
  for (Index_t i = 0; i < registers_per_block_; ++i) {
    free_indices.emplace_back(i);
  }

  const auto lambda_node = [&] {
    const auto &derefd_node = *mem_pool_acc[0].derefHandle(lambda);
    if (derefd_node.m_type != Compiler::ASTNode::Type::Lambda) {
      throw std::invalid_argument("Unexpected node type, not a lambda");
    }
    return static_cast<const Compiler::LambdaNode &>(derefd_node);
  }();

  std::unordered_map<Index_t, Index_t> lambda_space_ident_to_register;
  std::set<Index_t> lambda_captured_indices;
  if (const auto block_iter = lambdas_to_blocks.find(lambda);
      block_iter != lambdas_to_blocks.end() && block_iter->second == 2) {
    volatile int x = 1;
    x = x + 1;
  }
  extract_lambda_space_captured_indices(lambda, 0, mem_pool_acc,
                                        lambda_captured_indices);
  std::unordered_map<Index_t, Index_t> lambda_space_ident_to_remaining_count;
  {
    // Captured values copied into beginning of registers, followed by args.
    for (const auto captured_index : lambda_captured_indices) {
      lambda_space_ident_to_register[captured_index] = free_indices.front();
      lambda_space_ident_to_remaining_count[captured_index] = 0;
      free_indices.pop_front();
    }
  }
  for (Index_t i = 0; i < lambda_node.m_argCount; ++i) {
    lambda_space_ident_to_remaining_count[i] = 0;
    lambda_space_ident_to_register[i] = free_indices.front();
    free_indices.pop_front();
  }

  Index_t num_bound_so_far = lambda_node.m_argCount;
  compute_lambda_space_ident_to_use_count(
      lambda_node.m_childExpr, num_bound_so_far, mem_pool_acc,
      lambda_space_ident_to_remaining_count);

  const auto convert_to_lambda_space_index = [&](const auto original_index) {
    return num_bound_so_far - original_index - 1;
  };

  const auto lookup_register_for_ident = [&](const auto &maybe_ident) {
    if (maybe_ident.m_type != Compiler::ASTNode::Type::Identifier) {
      throw std::invalid_argument(
          "Cannot generate indirect call for target which is not an "
          "identifier. Indicates a bug in block preparation layer.");
    }
    const auto &ident_node =
        static_cast<const Compiler::IdentifierNode &>(maybe_ident);
    const auto ident_in_lambda_space =
        convert_to_lambda_space_index(ident_node.m_index);
    const auto register_idx_iter =
        lambda_space_ident_to_register.find(ident_in_lambda_space);
    if (register_idx_iter == lambda_space_ident_to_register.end()) {
      for (const auto idx : lambda_captured_indices) {
        std::cout << "captured idx: " << idx << std::endl;
      }
      throw std::invalid_argument("Failed to find register for identifier");
    }
    return register_idx_iter->second;
  };

  Index_t idents_mapped_so_far = lambda_node.m_argCount;
  const auto allocate_register = [&] {
    const auto allocated_idx = free_indices.front();
    free_indices.pop_front();
    const auto lambda_space_ident = idents_mapped_so_far++;
    if (lambda_space_ident_to_register.find(lambda_space_ident) !=
        lambda_space_ident_to_register.end()) {
      throw std::invalid_argument(
          "Attempted to bind identifier multiple times");
    }
    lambda_space_ident_to_register[lambda_space_ident] = allocated_idx;
    return allocated_idx;
  };

  std::vector<Instruction> result_instructions;
  const auto primitive_expression_to_instruction =
      [&](const Compiler::ASTNodeHandle ast_node,
          const std::optional<Index_t> pre_allocated_register,
          bool is_tail_instruction) {
        Instruction result;
        const auto allocate_register_or_use_pre_allocated = [&] {
          if (pre_allocated_register.has_value()) {
            return *pre_allocated_register;
          }
          return allocate_register();
        };
        visit(
            *mem_pool_acc[0].derefHandle(ast_node),
            Visitor{
                [&](const Compiler::CallNode &call_node) {
                  result.type = is_tail_instruction
                                    ? InstructionType::BLOCKING_CALL_INDIRECT
                                    : InstructionType::CALL_INDIRECT;
                  const auto &target_ast_node =
                      *mem_pool_acc[0].derefHandle(call_node.m_target);
                  result.data.call_indirect.lambda_idx =
                      lookup_register_for_ident(target_ast_node);
                  const auto *args_data =
                      mem_pool_acc[0].derefHandle(call_node.m_args);
                  const auto arg_indices = mem_pool_acc[0].AllocArray<Index_t>(
                      call_node.m_args.GetCount());
                  auto *arg_indices_data =
                      mem_pool_acc[0].derefHandle(arg_indices);
                  for (Index_t arg_idx = 0; arg_idx < arg_indices.GetCount();
                       ++arg_idx) {
                    const auto &arg_elem =
                        *mem_pool_acc[0].derefHandle(args_data[arg_idx]);
                    arg_indices_data[arg_idx] =
                        lookup_register_for_ident(arg_elem);
                  }
                  const auto assign_call_indirect_fields =
                      [&](CallIndirectCommon &derived_result) {
                        derived_result.arg_indices = arg_indices;
                        derived_result.target_register =
                            allocate_register_or_use_pre_allocated();
                      };
                  visit(result,
                        Visitor{[&](CallIndirect &call_indirect) {
                                  assign_call_indirect_fields(call_indirect);
                                },
                                [&](BlockingCallIndirect &blocking_indirect) {
                                  assign_call_indirect_fields(
                                      blocking_indirect);
                                },
                                [](auto &) {
                                  throw std::invalid_argument("Unexpected");
                                }},
                        [](auto &) {
                          throw std::invalid_argument("Unexpected");
                        });
                },
                [&](const Compiler::LambdaNode &) {
                  result.type = InstructionType::CREATE_LAMBDA;
                  const auto lambda_to_block_idx_iter =
                      lambdas_to_blocks.find(ast_node);
                  if (lambda_to_block_idx_iter == lambdas_to_blocks.end()) {
                    throw std::invalid_argument(
                        "Lambda to block idx unspecified for current lambda "
                        "node");
                  }
                  result.data.create_lambda.block_idx =
                      lambda_to_block_idx_iter->second;
                  if (lambda_to_block_idx_iter->second == 2) {
                    volatile int x = 1;
                    x = x + 1;
                  }
                  const auto captured_indices = [&] {
                    std::set<Index_t> captured_indices_set;
                    extract_lambda_space_captured_indices(
                        ast_node, 0, mem_pool_acc, captured_indices_set);
                    return std::vector<Index_t>(captured_indices_set.begin(),
                                                captured_indices_set.end());
                  }();
                  const auto captured_indices_handle =
                      mem_pool_acc[0].AllocArray<Index_t>(
                          captured_indices.size());
                  auto *captured_indices_data =
                      mem_pool_acc[0].derefHandle(captured_indices_handle);
                  for (Index_t i = 0; i < captured_indices_handle.GetCount();
                       ++i) {
                    const auto index_in_this_lambda =
                        captured_indices[i] + num_bound_so_far;
                    const auto register_idx_iter =
                        lambda_space_ident_to_register.find(
                            index_in_this_lambda);
                    if (register_idx_iter ==
                        lambda_space_ident_to_register.end()) {
                      throw std::invalid_argument("Failed to find register for "
                                                  "captured value in nested "
                                                  "lambda");
                    }
                    captured_indices_data[i] = register_idx_iter->second;
                  }
                  result.data.create_lambda.captured_indices =
                      captured_indices_handle;
                  result.data.create_lambda.target_register =
                      allocate_register_or_use_pre_allocated();
                },
                [&](const Compiler::IfNode &if_node) {
                  result.type = InstructionType::IF;
                  const auto &predicate_node =
                      *mem_pool_acc[0].derefHandle(if_node.m_pred);
                  result.data.if_val.predicate =
                      lookup_register_for_ident(predicate_node);
                  result.data.if_val.goto_true = result_instructions.size() + 1;
                  result.data.if_val.goto_false =
                      result_instructions.size() + 2;
                },
                [&](const Compiler::NumberNode &number_node) {
                  result.type = InstructionType::ASSIGN_CONSTANT;
                  result.data.assign_constant.constant = number_node.m_value;
                  result.data.assign_constant.target_register =
                      allocate_register_or_use_pre_allocated();
                },
                [&](const Compiler::BinaryOpNode &binary_op_node) {
                  result.type = [&]() -> InstructionType {
                    switch (binary_op_node.m_type) {
                    case Compiler::ASTNode::Type::Add:
                      return InstructionType::ADD;
                    case Compiler::ASTNode::Type::Sub:
                      return InstructionType::SUB;
                    case Compiler::ASTNode::Type::Mul:
                      return InstructionType::MUL;
                    case Compiler::ASTNode::Type::Div:
                      return InstructionType::DIV;
                    case Compiler::ASTNode::Type::Equal:
                      return InstructionType::EQUAL;
                    case Compiler::ASTNode::Type::GreaterThan:
                      return InstructionType::GREATER_THAN;
                    case Compiler::ASTNode::Type::Remainder:
                      return InstructionType::REMAINDER;
                    case Compiler::ASTNode::Type::Expt:
                      return InstructionType::EXPT;
                    case Compiler::ASTNode::Type::Bind:
                    case Compiler::ASTNode::Type::BindRec:
                    case Compiler::ASTNode::Type::Call:
                    case Compiler::ASTNode::Type::If:
                    case Compiler::ASTNode::Type::Floor:
                    case Compiler::ASTNode::Type::Number:
                    case Compiler::ASTNode::Type::Identifier:
                    case Compiler::ASTNode::Type::Lambda:
                      throw std::invalid_argument(
                          "Unexpected AST node type for binary op");
                    }
                  }();
                  visit(result,
                        Visitor{[&]<InstructionType TYPE>(
                                    BinaryOp<TYPE> &instruction) {
                                  instruction.lhs = lookup_register_for_ident(
                                      *mem_pool_acc[0].derefHandle(
                                          binary_op_node.m_arg0));
                                  instruction.rhs = lookup_register_for_ident(
                                      *mem_pool_acc[0].derefHandle(
                                          binary_op_node.m_arg1));
                                  instruction.target_register =
                                      allocate_register_or_use_pre_allocated();
                                },
                                [&](auto &elem) {
                                  std::cout
                                      << "Mapped result type: "
                                      << static_cast<uint64_t>(result.type)
                                      << std::endl;
                                  throw std::invalid_argument(
                                      "Unexpected binary op");
                                }},
                        [](auto &) {
                          throw std::invalid_argument(
                              "Unexpected instruction type");
                        });
                },
                [&](const Compiler::UnaryOpNode &unary_op_node) {
                  result.type = [&]() -> InstructionType {
                    switch (unary_op_node.m_type) {
                    case Compiler::ASTNode::Type::Floor:
                      return InstructionType::FLOOR;
                    default:
                      throw std::invalid_argument("Unexpected unary op node");
                    }
                  }();
                  visit(result,
                        Visitor{[&](Floor &floor) {
                                  floor.arg = lookup_register_for_ident(
                                      *mem_pool_acc[0].derefHandle(
                                          unary_op_node.m_arg0));
                                  floor.target_register =
                                      allocate_register_or_use_pre_allocated();
                                },
                                [&](const auto &) {
                                  throw std::invalid_argument(
                                      "Unexpected type");
                                }},
                        [](const auto &) {
                          throw std::invalid_argument(
                              "Unexpected instruction type");
                        });
                },
                [&](const Compiler::BindNode &) {
                  throw std::invalid_argument(
                      "Should not generate instructions for bind nodes");
                },
                [&](const Compiler::IdentifierNode &ident_node) {
                  result.type = InstructionType::ASSIGN;
                  result.data.assign.target_register =
                      allocate_register_or_use_pre_allocated();
                  result.data.assign.source_register =
                      lookup_register_for_ident(ident_node);
                }},
            [](const auto &) {
              throw std::invalid_argument("Unexpected compiled node");
            });

        auto idents_in_most_recent_expression = [&] {
          const auto &derefd_ast_node = *mem_pool_acc[0].derefHandle(ast_node);
          if (derefd_ast_node.m_type == Compiler::ASTNode::Type::Lambda) {
            std::set<Index_t> captured_indices_set;
            extract_lambda_space_captured_indices(ast_node, 0, mem_pool_acc,
                                                  captured_indices_set);
            std::set<Index_t> result;
            for (const auto captured_index : captured_indices_set) {
              result.emplace(captured_index + num_bound_so_far);
            }
            return result;
          }
          const auto ast_node_to_obtain_idents_from = [&] {
            if (derefd_ast_node.m_type != Compiler::ASTNode::Type::If) {
              return ast_node;
            }
            const auto &if_node =
                static_cast<const Compiler::IfNode &>(derefd_ast_node);
            return if_node.m_pred;
          }();
          std::set<const Compiler::ASTNodeHandle *> identifiers;
          CollectAllASTNodes(ast_node_to_obtain_idents_from, mem_pool_acc,
                             {Compiler::ASTNode::Type::Identifier},
                             identifiers);
          std::set<Index_t> result;
          for (const auto ident : identifiers) {
            const auto &ident_node = *mem_pool_acc[0].derefHandle(*ident);
            if (ident_node.m_type != Compiler::ASTNode::Type::Identifier) {
              throw std::invalid_argument("Expected only identifiers in "
                                          "collection from CollectAllASTNodes");
            }
            result.emplace(convert_to_lambda_space_index(
                static_cast<const Compiler::IdentifierNode &>(ident_node)
                    .m_index));
          }
          return result;
        }();

        for (const auto ident_in_lambda_space :
             idents_in_most_recent_expression) {
          auto iter =
              lambda_space_ident_to_remaining_count.find(ident_in_lambda_space);
          if (iter == lambda_space_ident_to_remaining_count.end()) {
            throw std::invalid_argument(
                "Ident referenced after remaining count reached 0");
          }
          --iter->second;
          if (iter->second == 0) {
            const auto bound_register_iter =
                lambda_space_ident_to_register.find(ident_in_lambda_space);
            if (bound_register_iter == lambda_space_ident_to_register.end()) {
              throw std::invalid_argument(
                  "Register with previously non-zero count not bound");
            }
            free_indices.emplace_back(bound_register_iter->second);
            lambda_space_ident_to_register.erase(bound_register_iter);
            lambda_space_ident_to_remaining_count.erase(iter);
          }
        }

        return result;
      };

  Compiler::ASTNodeHandle pending_generation = lambda_node.m_childExpr;
  while (pending_generation != Compiler::ASTNodeHandle()) {
    const auto *child_expr = mem_pool_acc[0].derefHandle(pending_generation);
    switch (child_expr->m_type) {
    case Compiler::ASTNode::Type::Bind:
    case Compiler::ASTNode::Type::BindRec: {
      const auto &bind_node =
          static_cast<const Compiler::BindNode &>(*child_expr);
      const auto *bound_expr_data =
          mem_pool_acc[0].derefHandle(bind_node.m_bindings);
      const auto is_rec =
          child_expr->m_type == Compiler::ASTNode::Type::BindRec;
      std::deque<Index_t> pre_allocated_registers;
      if (is_rec) {
        for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
          pre_allocated_registers.emplace_back(allocate_register());
        }
        num_bound_so_far += pre_allocated_registers.size();
      }
      auto bindingRequiresIndirectCall = false;
      for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
        std::optional<Index_t> pre_allocated_register;
        if (is_rec) {
          pre_allocated_register = pre_allocated_registers[i];
        }
        const auto &instruction = result_instructions.emplace_back(
            primitive_expression_to_instruction(bound_expr_data[i],
                                                pre_allocated_register, false));
        bindingRequiresIndirectCall =
            bindingRequiresIndirectCall ||
            instruction.type == InstructionType::CALL_INDIRECT;
      }
      if (bindingRequiresIndirectCall) {
        result_instructions.emplace_back(
            Instruction(InstructionType::INSTRUCTION_BARRIER));
      }
      pending_generation = bind_node.m_childExpr;
      if (!is_rec) {
        num_bound_so_far += bind_node.m_bindings.GetCount();
      }
      break;
    }
    case Compiler::ASTNode::Type::If: {
      const auto &if_node = static_cast<const Compiler::IfNode &>(*child_expr);
      result_instructions.emplace_back(primitive_expression_to_instruction(
          pending_generation, std::nullopt, false));
      result_instructions.emplace_back(primitive_expression_to_instruction(
          if_node.m_then, std::nullopt, true));
      result_instructions.emplace_back(primitive_expression_to_instruction(
          if_node.m_else, std::nullopt, true));
      // If exprs are always in tail position
      pending_generation = Compiler::ASTNodeHandle();
      break;
    }
    default:
      // Tail instruction
      auto &tail_instruction =
          result_instructions.emplace_back(primitive_expression_to_instruction(
              pending_generation, std::nullopt, true));
      pending_generation = Compiler::ASTNodeHandle();
      break;
    }
  }
  const auto instructions_handle =
      mem_pool_acc[0].AllocArray<Instruction>(result_instructions.size());
  auto *instructions_data = mem_pool_acc[0].derefHandle(instructions_handle);
  for (Index_t i = 0; i < instructions_handle.GetCount(); ++i) {
    instructions_data[i] = result_instructions[i];
  }
  return Lambda(instructions_handle);
}

Program BlockGenerator::construct_blocks(const Compiler::ASTNodeHandle root) {
  auto mem_pool_acc =
      pool_.template get_access<cl::sycl::access::mode::read_write>();
  Index_t initial_index = 0;
  // TODO this construction makes all lambdas recursive. Maybe only make
  // the lambda index available in the map after the block has been generated if
  // the lambda was not bound recursively.
  std::map<Compiler::ASTNodeHandle, Index_t> lambdas_to_blocks;
  assign_lambdas_block_indices(mem_pool_acc, root, lambdas_to_blocks,
                               initial_index);
  const auto all_lambdas_handle =
      mem_pool_acc[0].AllocArray<Lambda>(lambdas_to_blocks.size());
  auto *all_lambdas_data = mem_pool_acc[0].derefHandle(all_lambdas_handle);
  for (const auto &[lambda, block_idx] : lambdas_to_blocks) {
    all_lambdas_data[block_idx] =
        construct_block(lambda, root, lambdas_to_blocks, mem_pool_acc);
  }

  return all_lambdas_handle;
}
} // namespace FunGPU::EvaluatorV2
