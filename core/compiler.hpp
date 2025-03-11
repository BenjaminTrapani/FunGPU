#pragma once

#include "core/array.hpp"
#include "core/ast_node.hpp"
#include "core/portable_mem_pool.hpp"
#include "core/s_expr.hpp"
#include <CL/sycl.hpp>

#include <list>
#include <memory>

namespace FunGPU {
class Compiler {
public:
  class CompileException {
  public:
    CompileException(const std::string &what) : m_what(what) {}
    const std::string &what() const { return m_what; }

  private:
    std::string m_what;
  };

  struct CompileResult {
    ASTNodeHandle ast_root;
    std::vector<std::string> all_identifiers;
  };

  Compiler(std::shared_ptr<const SExpr> sexpr,
           cl::sycl::buffer<PortableMemPool> pool)
      : m_sexpr(sexpr), m_mem_pool(pool) {}

  CompileResult compile() {
    std::list<std::string> initial_bound;
    auto mem_pool_acc =
        m_mem_pool.get_access<cl::sycl::access::mode::read_write>();
    std::vector<std::string> all_identifiers;
    const auto ast_root =
        compile(m_sexpr, initial_bound, all_identifiers, mem_pool_acc);
    return CompileResult(ast_root, all_identifiers);
  }

  void debug_print_ast(ASTNodeHandle root_of_ast);
  void deallocate_ast(const ASTNodeHandle root_of_ast);

private:
  static void debug_print_ast(ASTNodeHandle root_of_ast,
                              PortableMemPool::HostAccessor_t mem_pool_acc,
                              std::size_t indent);
  static void deallocate_ast(const ASTNodeHandle &handle,
                             PortableMemPool::HostAccessor_t mem_pool_acc);

  static ASTNodeHandle compile(std::shared_ptr<const SExpr> sexpr,
                               std::list<std::string> bound_identifiers,
                               std::vector<std::string> &all_identifiers,
                               PortableMemPool::HostAccessor_t mem_pool_acc);
  static ASTNodeHandle
  compile_list_of_sexpr(std::shared_ptr<const SExpr> sexpr,
                        std::list<std::string> bound_identifiers,
                        std::vector<std::string> &all_identifiers,
                        PortableMemPool::HostAccessor_t mem_pool_acc);

  std::shared_ptr<const SExpr> m_sexpr;
  cl::sycl::buffer<PortableMemPool> m_mem_pool;
};
} // namespace FunGPU
