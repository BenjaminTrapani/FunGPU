#pragma once

#include "core/ast_node.hpp"
#include <string>

namespace FunGPU {
std::string
serialize_ast_as_fgpu_program(const ASTNodeHandle &,
                              const std::vector<std::string> &all_identifiers,
                              PortableMemPool::HostAccessor_t);
}
