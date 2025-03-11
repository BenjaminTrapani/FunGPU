#pragma once

#include "core/ast_node.hpp"
#include "core/portable_mem_pool.hpp"
#include <set>

namespace FunGPU {
void collect_all_ast_nodes(ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t mem_pool_acc,
                           const std::set<ASTNode::Type> &types,
                           std::set<ASTNodeHandle *> &result);

void collect_all_ast_nodes(const ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t mem_pool_acc,
                           const std::set<ASTNode::Type> &types,
                           std::set<const ASTNodeHandle *> &result);
} // namespace FunGPU
