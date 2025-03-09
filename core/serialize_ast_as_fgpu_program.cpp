#include "core/serialize_ast_as_fgpu_program.hpp"
#include "core/compiler.hpp"
#include "core/visitor.hpp"
#include <deque>
#include <ostream>
#include <span>

namespace FunGPU {
namespace {
constexpr std::size_t INDENT_SIZE = 2;

void serialize_ast_as_fgpu_program(
    const Compiler::ASTNodeHandle &ast,
    PortableMemPool::HostAccessor_t mem_pool_acc, std::ostream &os,
    const std::size_t indent, const std::vector<std::string> &all_identifiers,
    std::size_t &identifier_index, std::deque<std::string> &bound_identifiers) {
  const auto &ast_node = *mem_pool_acc[0].deref_handle(ast);
  const auto newline_str = "\n" + std::string(indent, ' ');
  const auto single_indent_str = std::string(INDENT_SIZE, ' ');
  visit(
      ast_node,
      Visitor{
          [&](const Compiler::BindNode &node) {
            os << (node.is_rec() ? "(letrec " : "(let ") << "( ";
            const auto *binding_data =
                mem_pool_acc[0].deref_handle(node.m_bindings);
            const std::span<const Compiler::ASTNodeHandle> bindings_span(
                binding_data, node.m_bindings.get_count());
            const auto original_identifier_index = identifier_index;
            const auto push_all_bound_identifiers = [&] {
              for (std::size_t i = 0; i < bindings_span.size(); ++i) {
                bound_identifiers.push_back(
                    all_identifiers.at(original_identifier_index + i));
              }
            };
            if (node.is_rec()) {
              push_all_bound_identifiers();
            }
            const std::span binding_span(binding_data,
                                         node.m_bindings.get_count());
            identifier_index += bindings_span.size();
            os << newline_str;
            for (std::size_t i = 0; i < binding_span.size(); ++i) {
              os << single_indent_str << "(";
              os << all_identifiers.at(original_identifier_index + i) << " ";
              serialize_ast_as_fgpu_program(
                  binding_span[i], mem_pool_acc, os, indent + INDENT_SIZE,
                  all_identifiers, identifier_index, bound_identifiers);
              os << ")" << newline_str;
            }
            os << ")" << newline_str << single_indent_str;
            if (!node.is_rec()) {
              push_all_bound_identifiers();
            }
            serialize_ast_as_fgpu_program(node.m_child_expr, mem_pool_acc, os,
                                          indent + INDENT_SIZE, all_identifiers,
                                          identifier_index, bound_identifiers);
            for (std::size_t i = 0; i < bindings_span.size(); ++i) {
              bound_identifiers.pop_back();
            }
            os << ")";
          },
          [&](const Compiler::LambdaNode &node) {
            os << "(lambda (";
            for (std::size_t i = 0; i < node.m_arg_count; ++i) {
              const auto &bound_identifier =
                  all_identifiers.at(identifier_index + i);
              os << bound_identifier;
              if (i < node.m_arg_count - 1) {
                os << " ";
              }
              bound_identifiers.push_back(bound_identifier);
            }
            os << ")" << newline_str << single_indent_str;
            identifier_index += node.m_arg_count;
            serialize_ast_as_fgpu_program(node.m_child_expr, mem_pool_acc, os,
                                          indent + INDENT_SIZE, all_identifiers,
                                          identifier_index, bound_identifiers);
            for (std::size_t i = 0; i < node.m_arg_count; ++i) {
              bound_identifiers.pop_back();
            }
            os << ")";
          },
          [&](const Compiler::NumberNode &node) { os << node.m_value; },
          [&](const Compiler::IdentifierNode &node) {
            const auto ident_index = bound_identifiers.size() -
                                     static_cast<std::size_t>(node.m_index) -
                                     1UZ;
            if (ident_index >= bound_identifiers.size()) {
              throw std::runtime_error(
                  "Identifier index out of bounds: " +
                  std::to_string(node.m_index) + " with bound identifiers: " +
                  std::to_string(bound_identifiers.size()));
            }
            os << bound_identifiers.at(ident_index);
          },
          [&](const Compiler::CallNode &node) {
            os << "(";
            serialize_ast_as_fgpu_program(node.m_target, mem_pool_acc, os,
                                          indent, all_identifiers,
                                          identifier_index, bound_identifiers);
            const auto *arg_data = mem_pool_acc[0].deref_handle(node.m_args);
            const std::span<const Compiler::ASTNodeHandle> args_span(
                arg_data, node.m_args.get_count());
            for (const auto &arg : args_span) {
              os << " ";
              serialize_ast_as_fgpu_program(arg, mem_pool_acc, os, indent,
                                            all_identifiers, identifier_index,
                                            bound_identifiers);
            }
            os << ")";
          },
          [&](const Compiler::IfNode &node) {
            os << "(if ";
            serialize_ast_as_fgpu_program(node.m_pred, mem_pool_acc, os, indent,
                                          all_identifiers, identifier_index,
                                          bound_identifiers);
            os << newline_str << single_indent_str;
            serialize_ast_as_fgpu_program(node.m_then, mem_pool_acc, os,
                                          indent + INDENT_SIZE, all_identifiers,
                                          identifier_index, bound_identifiers);
            os << newline_str << single_indent_str;
            serialize_ast_as_fgpu_program(node.m_else, mem_pool_acc, os,
                                          indent + INDENT_SIZE, all_identifiers,
                                          identifier_index, bound_identifiers);
            os << ")";
          },
          [&](const Compiler::UnaryOpNode &node) {
            os << "(";
            switch (node.node_type) {
            case Compiler::ASTNode::Type::Floor:
              os << "floor ";
              break;
            default:
              throw std::runtime_error(
                  "Unexpected unary operator" +
                  std::to_string(static_cast<int>(node.node_type)));
            }
            serialize_ast_as_fgpu_program(node.m_arg0, mem_pool_acc, os, indent,
                                          all_identifiers, identifier_index,
                                          bound_identifiers);
            os << ")";
          },
          [&](const Compiler::BinaryOpNode &node) {
            os << "(";
            const std::string_view op_str = [&] -> std::string_view {
              switch (node.node_type) {
              case Compiler::ASTNode::Type::Add:
                return "+";
              case Compiler::ASTNode::Type::Sub:
                return "-";
              case Compiler::ASTNode::Type::Mul:
                return "*";
              case Compiler::ASTNode::Type::Div:
                return "/";
              case Compiler::ASTNode::Type::Equal:
                return "=";
              case Compiler::ASTNode::Type::GreaterThan:
                return ">";
              case Compiler::ASTNode::Type::Remainder:
                return "remainder";
              case Compiler::ASTNode::Type::Expt:
                return "expt";
              case Compiler::ASTNode::Type::Bind:
              case Compiler::ASTNode::Type::BindRec:
              case Compiler::ASTNode::Type::Call:
              case Compiler::ASTNode::Type::If:
              case Compiler::ASTNode::Type::Lambda:
              case Compiler::ASTNode::Type::Number:
              case Compiler::ASTNode::Type::Identifier:
              case Compiler::ASTNode::Type::Floor:
                break;
              }
              throw std::runtime_error(
                  "Unexpected binary operator" +
                  std::to_string(static_cast<int>(node.node_type)));
            }();
            os << op_str << " ";
            serialize_ast_as_fgpu_program(node.m_arg0, mem_pool_acc, os, indent,
                                          all_identifiers, identifier_index,
                                          bound_identifiers);
            os << " ";
            serialize_ast_as_fgpu_program(node.m_arg1, mem_pool_acc, os, indent,
                                          all_identifiers, identifier_index,
                                          bound_identifiers);
            os << ")";
          },
      },
      [](const auto &unknown_node) {
        throw std::runtime_error(
            "Unexpected AST node type" +
            std::to_string(static_cast<int>(unknown_node.node_type)));
      });
}
} // namespace

std::string
serialize_ast_as_fgpu_program(const Compiler::ASTNodeHandle &ast,
                              const std::vector<std::string> &all_identifiers,
                              PortableMemPool::HostAccessor_t mem_pool_acc) {
  std::stringstream ss;
  std::deque<std::string> bound_identifiers;
  std::size_t identifier_index = 0;
  serialize_ast_as_fgpu_program(ast, mem_pool_acc, ss, 0, all_identifiers,
                                identifier_index, bound_identifiers);
  return ss.str();
}
} // namespace FunGPU
