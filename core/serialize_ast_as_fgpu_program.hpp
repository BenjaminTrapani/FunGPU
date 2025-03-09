#pragma once

#include "core/compiler.hpp"
#include <string>

namespace FunGPU {
std::string
serialize_ast_as_fgpu_program(const Compiler::ASTNodeHandle &,
                              const std::vector<std::string> &all_identifiers,
                              PortableMemPool::HostAccessor_t);
}
