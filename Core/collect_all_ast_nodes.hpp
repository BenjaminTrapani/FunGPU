#pragma once

#include "Core/compiler.hpp"
#include "Core/portable_mem_pool.hpp"
#include <set>

namespace FunGPU {
void collect_all_ast_nodes(Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t mem_pool_acc,
                           const std::set<Compiler::ASTNode::Type> &types,
                           std::set<Compiler::ASTNodeHandle *> &result);

void collect_all_ast_nodes(const Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t mem_pool_acc,
                           const std::set<Compiler::ASTNode::Type> &types,
                           std::set<const Compiler::ASTNodeHandle *> &result);
} // namespace FunGPU
