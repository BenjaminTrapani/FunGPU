#include "Core/CollectAllASTNodes.hpp"

namespace FunGPU {
  void CollectAllASTNodes(Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t memPoolAcc,
                           const std::set<Compiler::ASTNode::Type> &types,
                           std::set<Compiler::ASTNodeHandle *> &result) {
  visit(
      *memPoolAcc[0].derefHandle(root),
      [&](auto &node) {
        if (types.find(node.m_type) != types.end()) {
          result.emplace(&root);
          return;
        }
        node.for_each_sub_expr(memPoolAcc, [&](auto &child_expr) {
          CollectAllASTNodes(child_expr, memPoolAcc, types, result);
        });
      },
      [](const auto &unexpected) {
        throw std::invalid_argument("Unexpected node");
      });
}

void CollectAllASTNodes(const Compiler::ASTNodeHandle &root,
                        PortableMemPool::HostAccessor_t memPoolAcc,
                          const std::set<Compiler::ASTNode::Type> &types,
                          std::set<const Compiler::ASTNodeHandle *> &result) {
  std::set<Compiler::ASTNodeHandle*> mutable_result;
  CollectAllASTNodes(const_cast<Compiler::ASTNodeHandle&>(root), memPoolAcc, types, result);
  for (const auto handle : mutable_result) {
    result.emplace(handle);
  }
}
}
