#pragma once

#pragma once

#include "Core/Compiler.hpp"
#include "Core/PortableMemPool.hpp"
#include <set>

namespace FunGPU {
void CollectAllASTNodes(Compiler::ASTNodeHandle &root,
                        PortableMemPool::HostAccessor_t memPoolAcc,
                        const std::set<Compiler::ASTNode::Type> &types,
                        std::set<Compiler::ASTNodeHandle *> &result);

void CollectAllASTNodes(const Compiler::ASTNodeHandle &root,
                        PortableMemPool::HostAccessor_t memPoolAcc,
                        const std::set<Compiler::ASTNode::Type> &types,
                        std::set<const Compiler::ASTNodeHandle *> &result);
} // namespace FunGPU
