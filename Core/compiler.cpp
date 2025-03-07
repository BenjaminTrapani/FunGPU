#include "Core/compiler.hpp"
#include "Core/types.hpp"
#include <iostream>
#include <sstream>

namespace FunGPU {
Compiler::ASTNodeHandle
Compiler::compile_list_of_sexpr(std::shared_ptr<const SExpr> sexpr,
                                std::list<std::string> bound_identifiers,
                                PortableMemPool::HostAccessor_t mem_pool_acc) {
  auto sexpr_children = sexpr->get_children();
  if (sexpr_children->size() < 1) {
    throw CompileException("List of sexpr is less than 1, invalid expr");
  }

  auto first_child = sexpr_children->at(0);
  ASTNodeHandle result;
  if (first_child->get_type() == SExpr::Type::Symbol) {
    const auto first_child_sym = *first_child->get_symbol();
    if (first_child_sym == "+") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected + to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Add,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "-") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected - to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Sub,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "*") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected * to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Mul,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "/") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected / to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Div,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "=" || first_child_sym == "eq?") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected = to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Equal,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == ">") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected > to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::GreaterThan,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "remainder") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected remainder to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Remainder,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "expt") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected expt to have 2 args");
      }
      result = mem_pool_acc[0].template alloc<BinaryOpNode>(
          ASTNode::Type::Expt,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc),
          compile(sexpr_children->at(2), bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "let" || first_child_sym == "letrec") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected let to have 2 args");
      }
      const bool is_rec = first_child_sym == "letrec";

      auto binding_exprs = sexpr_children->at(1)->get_children();
      auto expr_to_eval_in_binding_env = sexpr_children->at(2);
      const auto bind_node_handle = mem_pool_acc[0].template alloc<BindNode>(
          static_cast<Index_t>(binding_exprs->size()), is_rec, mem_pool_acc);
      auto bind_node = mem_pool_acc[0].deref_handle(bind_node_handle);

      auto updated_bindings = bound_identifiers;
      for (const auto &bind_expr : *binding_exprs) {
        auto bind_expr_children = bind_expr->get_children();
        if (bind_expr_children->size() != 2) {
          throw CompileException("Expected binding expr to have 2 elements "
                                 "(identifier, value expr)");
        }
        auto ident_expr = bind_expr_children->at(0);
        if (ident_expr->get_type() != SExpr::Type::Symbol) {
          throw CompileException(
              "Expected first component of bind expr to be an identifier");
        }

        const auto ident_string = ident_expr->get_symbol();
        updated_bindings.push_front(*ident_string);
      }

      auto bindings_data = mem_pool_acc[0].deref_handle(bind_node->m_bindings);
      for (Index_t i = 0; i < binding_exprs->size(); ++i) {
        auto bind_expr = binding_exprs->at(i);
        auto bind_expr_children = bind_expr->get_children();
        bindings_data[i] = compile(
            bind_expr_children->at(1),
            is_rec ? updated_bindings : bound_identifiers, mem_pool_acc);
      }

      bind_node->m_child_expr =
          compile(sexpr_children->at(2), updated_bindings, mem_pool_acc);
      result = bind_node_handle;
    } else if (first_child_sym == "lambda") {
      if (sexpr_children->size() != 3) {
        throw CompileException("Expected lambda to have 2 child exprs");
      }
      auto identifier_list = sexpr_children->at(1);
      if (identifier_list->get_type() != SExpr::Type::ListOfSExpr) {
        throw CompileException("Lambda arg list expected to be list of sexpr");
      }
      auto identifier_list_children = identifier_list->get_children();
      for (auto identifier : *identifier_list_children) {
        if (identifier->get_type() != SExpr::Type::Symbol) {
          throw CompileException(
              "Expected arguments in lambda expression to be symbols");
        }
        bound_identifiers.push_front(*identifier->get_symbol());
      }
      auto expr_to_eval = sexpr_children->at(2);
      auto compiled_ast_node =
          compile(expr_to_eval, bound_identifiers, mem_pool_acc);
      result = mem_pool_acc[0].template alloc<LambdaNode>(
          static_cast<Index_t>(identifier_list_children->size()),
          compiled_ast_node);
    } else if (first_child_sym == "if") {
      if (sexpr_children->size() != 4) {
        throw CompileException("Expected if expr to have 3 arguments");
      }
      auto pred_child = sexpr_children->at(1);
      auto then_child = sexpr_children->at(2);
      auto else_child = sexpr_children->at(3);
      result = mem_pool_acc[0].template alloc<IfNode>(
          compile(pred_child, bound_identifiers, mem_pool_acc),
          compile(then_child, bound_identifiers, mem_pool_acc),
          compile(else_child, bound_identifiers, mem_pool_acc));
    } else if (first_child_sym == "floor") {
      if (sexpr_children->size() != 2) {
        throw CompileException("Expected floor to get 1 argument");
      }
      result = mem_pool_acc[0].template alloc<UnaryOpNode>(
          ASTNode::Type::Floor,
          compile(sexpr_children->at(1), bound_identifiers, mem_pool_acc));
    }
  }
  if (result ==
      ASTNodeHandle()) // This is hopefully a call to user-defined function.
  {
    auto arg_count = sexpr_children->size() - 1;
    auto target_lambda_expr = sexpr_children->at(0);
    const auto call_node_handle = mem_pool_acc[0].template alloc<CallNode>(
        static_cast<Index_t>(arg_count),
        compile(target_lambda_expr, bound_identifiers, mem_pool_acc),
        mem_pool_acc);
    auto call_node = mem_pool_acc[0].deref_handle(call_node_handle);
    auto args_data = mem_pool_acc[0].deref_handle(call_node->m_args);
    for (Index_t i = 1; i < sexpr_children->size(); ++i) {
      auto cur_arg = sexpr_children->at(i);
      args_data[i - 1] = compile(cur_arg, bound_identifiers, mem_pool_acc);
    }
    result = call_node_handle;
  }

  if (result == ASTNodeHandle()) {
    throw CompileException("Failed to compile list of sexpr");
  }

  return result;
}

Compiler::ASTNodeHandle
Compiler::compile(std::shared_ptr<const SExpr> sexpr,
                  std::list<std::string> bound_identifiers,
                  PortableMemPool::HostAccessor_t mem_pool_acc) {
  ASTNodeHandle result;
  switch (sexpr->get_type()) {
  case SExpr::Type::Symbol: {
    auto ident_pos = std::find(bound_identifiers.begin(),
                               bound_identifiers.end(), *sexpr->get_symbol());
    if (ident_pos == bound_identifiers.end()) {
      std::stringstream sstream;
      sstream << "Unbound identifier " << *sexpr->get_symbol() << std::endl;
      throw CompileException(sstream.str());
    }
    result =
        mem_pool_acc[0].template alloc<IdentifierNode>(static_cast<Index_t>(
            std::distance(bound_identifiers.begin(), ident_pos)));
    break;
  }
  case SExpr::Type::Number: {
    result = mem_pool_acc[0].template alloc<NumberNode>(sexpr->get_float_val());
    break;
  }
  case SExpr::Type::ListOfSExpr: {
    result = compile_list_of_sexpr(sexpr, bound_identifiers, mem_pool_acc);
    break;
  }
  default:
    throw CompileException("Unexpected sexpr type");
    break;
  }

  if (result == ASTNodeHandle()) {
    throw CompileException("Failed to compile sexpr");
  }

  return result;
}

void Compiler::debug_print_ast(ASTNodeHandle rootOfASTHandle) {
  auto mem_pool_acc =
      m_mem_pool.get_access<cl::sycl::access::mode::read_write>();
  debug_print_ast(rootOfASTHandle, mem_pool_acc, 0);
}

void Compiler::debug_print_ast(ASTNodeHandle root_of_ast_handle,
                               PortableMemPool::HostAccessor_t mem_pool_acc,
                               std::size_t indent) {
  auto *root_of_ast = mem_pool_acc[0].deref_handle(root_of_ast_handle);

  const auto start_new_line = [&] {
    std::cout << std::endl;
    for (std::size_t i = 0; i < indent; ++i) {
      std::cout << " ";
    }
  };

  switch (root_of_ast->node_type) {
  case ASTNode::Type::Bind:
  case ASTNode::Type::BindRec: {
    auto *bind_node = static_cast<BindNode *>(root_of_ast);
    if (root_of_ast->node_type == ASTNode::Type::Bind) {
      std::cout << "(let ";
    } else {
      std::cout << "(letrec ";
    }
    std::cout << "(";
    auto bindings_data = mem_pool_acc[0].deref_handle(bind_node->m_bindings);
    for (Index_t i = 0; i < bind_node->m_bindings.get_count(); ++i) {
      start_new_line();
      debug_print_ast(bindings_data[i], mem_pool_acc, indent + 1);
    }
    std::cout << ")";
    start_new_line();
    debug_print_ast(bind_node->m_child_expr, mem_pool_acc, indent + 2);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::If: {
    std::cout << "(if ";
    auto *if_node = static_cast<IfNode *>(root_of_ast);
    debug_print_ast(if_node->m_pred, mem_pool_acc, indent + 1);
    start_new_line();
    debug_print_ast(if_node->m_then, mem_pool_acc, indent + 2);
    start_new_line();
    debug_print_ast(if_node->m_else, mem_pool_acc, indent + 2);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Add: {
    std::cout << "(+ ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Sub: {
    std::cout << "(- ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Mul: {
    std::cout << "(* ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Div: {
    std::cout << "(/ ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Equal: {
    std::cout << "(= ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::GreaterThan: {
    std::cout << "(> ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Remainder: {
    std::cout << "(remainder ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Expt: {
    std::cout << "(expt ";
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    debug_print_ast(binary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << " ";
    debug_print_ast(binary_op_node->m_arg1, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Floor: {
    std::cout << "(floor ";
    auto *unary_op_node = static_cast<UnaryOpNode *>(root_of_ast);
    debug_print_ast(unary_op_node->m_arg0, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Number: {
    auto *num_node = static_cast<NumberNode *>(root_of_ast);
    std::cout << num_node->m_value;
    break;
  }
  case ASTNode::Type::Identifier: {
    auto *ident_node = static_cast<IdentifierNode *>(root_of_ast);
    std::cout << "(ident: " << ident_node->m_index << ")";
    break;
  }
  case ASTNode::Type::Lambda: {
    auto *lambda_node = static_cast<LambdaNode *>(root_of_ast);
    std::cout << "(lambda (argCount: " << lambda_node->m_arg_count << ")";
    start_new_line();
    debug_print_ast(lambda_node->m_child_expr, mem_pool_acc, indent + 1);
    std::cout << ")";
    break;
  }
  case ASTNode::Type::Call: {
    auto *call_expr = static_cast<CallNode *>(root_of_ast);
    std::cout << "(call ";
    debug_print_ast(call_expr->m_target, mem_pool_acc, indent + 1);
    std::cout << " ";
    auto args_data = mem_pool_acc[0].deref_handle(call_expr->m_args);
    for (Index_t i = 0; i < call_expr->m_args.get_count(); ++i) {
      debug_print_ast(args_data[i], mem_pool_acc, indent + 1);
      std::cout << " ";
    }
    std::cout << ")";
    break;
  }
  default:
    throw CompileException("Unexpected AST node type during debug print");
  }
}

void Compiler::deallocate_ast(const ASTNodeHandle &root_of_ast_handle,
                              PortableMemPool::HostAccessor_t mem_pool_acc) {
  auto *root_of_ast = mem_pool_acc[0].deref_handle(root_of_ast_handle);
  switch (root_of_ast->node_type) {
  case ASTNode::Type::Bind:
  case ASTNode::Type::BindRec: {
    auto *bind_node = static_cast<BindNode *>(root_of_ast);
    auto *binding_data = mem_pool_acc[0].deref_handle(bind_node->m_bindings);
    for (Index_t i = 0; i < bind_node->m_bindings.get_count(); ++i) {
      deallocate_ast(binding_data[i], mem_pool_acc);
    }
    deallocate_ast(bind_node->m_child_expr, mem_pool_acc);
    mem_pool_acc[0].dealloc_array(bind_node->m_bindings);

    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<BindNode>>(root_of_ast_handle));
    break;
  }
  case ASTNode::Type::If: {
    auto *if_node = static_cast<IfNode *>(root_of_ast);
    deallocate_ast(if_node->m_pred, mem_pool_acc);
    deallocate_ast(if_node->m_then, mem_pool_acc);
    deallocate_ast(if_node->m_else, mem_pool_acc);

    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<IfNode>>(root_of_ast_handle));
    break;
  }
  case ASTNode::Type::Add:
  case ASTNode::Type::Sub:
  case ASTNode::Type::Mul:
  case ASTNode::Type::Div:
  case ASTNode::Type::Equal:
  case ASTNode::Type::GreaterThan:
  case ASTNode::Type::Remainder:
  case ASTNode::Type::Expt: {
    auto *binary_op_node = static_cast<BinaryOpNode *>(root_of_ast);
    deallocate_ast(binary_op_node->m_arg0, mem_pool_acc);
    deallocate_ast(binary_op_node->m_arg1, mem_pool_acc);

    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<BinaryOpNode>>(root_of_ast_handle));
    break;
  }
  case ASTNode::Type::Floor: {
    auto *unary_op_node = static_cast<UnaryOpNode *>(root_of_ast);
    deallocate_ast(unary_op_node->m_arg0, mem_pool_acc);

    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<UnaryOpNode>>(root_of_ast_handle));
    break;
  }
  case ASTNode::Type::Number:
    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<NumberNode>>(root_of_ast_handle));
    break;
  case ASTNode::Type::Identifier:
    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<IdentifierNode>>(
            root_of_ast_handle));
    break;
  case ASTNode::Type::Lambda: {
    auto *lambda_node = static_cast<LambdaNode *>(root_of_ast);
    deallocate_ast(lambda_node->m_child_expr, mem_pool_acc);

    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<LambdaNode>>(root_of_ast_handle));
    break;
  }
  case ASTNode::Type::Call: {
    auto *call_expr = static_cast<CallNode *>(root_of_ast);

    deallocate_ast(call_expr->m_target, mem_pool_acc);
    auto args_data = mem_pool_acc[0].deref_handle(call_expr->m_args);
    for (Index_t i = 0; i < call_expr->m_args.get_count(); ++i) {
      deallocate_ast(args_data[i], mem_pool_acc);
    }

    mem_pool_acc[0].dealloc_array(call_expr->m_args);

    mem_pool_acc[0].dealloc(
        static_cast<PortableMemPool::Handle<CallNode>>(root_of_ast_handle));
    break;
  }
  default:
    throw CompileException("Unexpected AST node type during deallocate");
  }
}

void Compiler::deallocate_ast(const ASTNodeHandle root_of_ast_handle) {
  auto mem_pool_acc =
      m_mem_pool.get_access<cl::sycl::access::mode::read_write>();
  deallocate_ast(root_of_ast_handle, mem_pool_acc);
}
} // namespace FunGPU
