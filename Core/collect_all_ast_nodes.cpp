#include "Core/collect_all_ast_nodes.hpp"

namespace FunGPU {
void collect_all_ast_nodes(Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t mem_pool_acc,
                           const std::set<Compiler::ASTNode::Type> &types,
                           std::set<Compiler::ASTNodeHandle *> &result) {
  visit(
      *mem_pool_acc[0].deref_handle(root),
      [&](auto &node) {
        if (types.find(node.node_type) != types.end()) {
          result.emplace(&root);
          return;
        }
        node.for_each_sub_expr(mem_pool_acc, [&](auto &child_expr) {
          collect_all_ast_nodes(child_expr, mem_pool_acc, types, result);
        });
      },
      [](const auto &unexpected) {
        throw std::invalid_argument("Unexpected node");
      });
}

void collect_all_ast_nodes(const Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t mem_pool_acc,
                           const std::set<Compiler::ASTNode::Type> &types,
                           std::set<const Compiler::ASTNodeHandle *> &result) {
  std::set<Compiler::ASTNodeHandle *> mutable_result;
  collect_all_ast_nodes(const_cast<Compiler::ASTNodeHandle &>(root),
                        mem_pool_acc, types, mutable_result);
  for (const auto handle : mutable_result) {
    result.emplace(handle);
  }
}
} // namespace FunGPU
