#pragma once

#include "Core/Array.hpp"
#include "Core/PortableMemPool.hpp"
#include "Core/SExpr.hpp"
#include "Core/Types.hpp"
#include <CL/sycl.hpp>

#include <list>
#include <memory>

namespace FunGPU {
class Compiler {
public:
  class ASTNode {
  public:
    enum class Type {
      Bind,
      BindRec,
      Call,
      If,
      Add,
      Sub,
      Mul,
      Div,
      Equal,
      GreaterThan,
      Remainder,
      Expt,
      Floor,
      Number,
      Identifier,
      Lambda,
    };
    ASTNode(Type type, Index_t frame_size)
        : node_type(type), frame_size(frame_size) {}

    Index_t get_frame_count() const { return frame_size; }

    Type node_type;
    Index_t frame_size;
  };

  using ASTNodeHandle = PortableMemPool::Handle<ASTNode>;

#define DEF_MUTABLE_FOR_EACH_SUB_EXPR(Type)                                    \
  template <typename CB>                                                       \
  void for_each_sub_expr(PortableMemPool::HostAccessor_t &host_acc, CB &&cb)   \
      const {                                                                  \
    const_cast<Type *>(this)->for_each_sub_expr(                               \
        host_acc, [&](auto &&elem) { cb(std::as_const(elem)); });              \
  }

  class BindNode : public ASTNode {
  public:
    BindNode(const Index_t num_bindings, const bool is_rec,
             PortableMemPool::HostAccessor_t pool)
        : ASTNode(is_rec ? Type::BindRec : Type::Bind, num_bindings),
          m_bindings(
              pool[0].template alloc_array<ASTNodeHandle>(num_bindings)) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t &host_acc, CB &&cb) {
      if (m_bindings != PortableMemPool::ArrayHandle<ASTNodeHandle>()) {
        auto *child_expr_data = host_acc[0].deref_handle(m_bindings);
        for (Index_t i = 0; i < m_bindings.get_count(); ++i) {
          cb(child_expr_data[i]);
        }
      }
      cb(m_child_expr);
    }

    bool is_rec() const noexcept { return node_type == Type::BindRec; }

    DEF_MUTABLE_FOR_EACH_SUB_EXPR(BindNode)

    PortableMemPool::ArrayHandle<ASTNodeHandle> m_bindings;
    ASTNodeHandle m_child_expr;
  };

  class UnaryOpNode : public ASTNode {
  public:
    UnaryOpNode(ASTNode::Type type, ASTNodeHandle arg0)
        : ASTNode(type, 1), m_arg0(arg0) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t &, CB &&cb) {
      cb(m_arg0);
    }

    DEF_MUTABLE_FOR_EACH_SUB_EXPR(UnaryOpNode)

    ASTNodeHandle m_arg0;
  };

  class BinaryOpNode : public ASTNode {
  public:
    BinaryOpNode(ASTNode::Type type, ASTNodeHandle arg0, ASTNodeHandle arg1)
        : ASTNode(type, 2), m_arg0(arg0), m_arg1(arg1) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t &, CB &&cb) {
      cb(m_arg0);
      cb(m_arg1);
    }

    DEF_MUTABLE_FOR_EACH_SUB_EXPR(BinaryOpNode)

    ASTNodeHandle m_arg0;
    ASTNodeHandle m_arg1;
  };

  class NumberNode : public ASTNode {
  public:
    NumberNode(const Float_t value)
        : ASTNode(ASTNode::Type::Number, 0), m_value(value) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t, CB &&) {}
    DEF_MUTABLE_FOR_EACH_SUB_EXPR(NumberNode)

    Float_t m_value;
  };

  class IdentifierNode : public ASTNode {
  public:
    IdentifierNode(const Index_t index)
        : ASTNode(ASTNode::Type::Identifier, 0), m_index(index) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t, CB &&) {}
    DEF_MUTABLE_FOR_EACH_SUB_EXPR(IdentifierNode)

    Index_t m_index;
  };

  class IfNode : public ASTNode {
  public:
    IfNode(ASTNodeHandle pred, ASTNodeHandle then, ASTNodeHandle elseExpr)
        : ASTNode(ASTNode::Type::If, 1), m_pred(pred), m_then(then),
          m_else(elseExpr) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t &, CB &&cb) {
      cb(m_pred);
      cb(m_then);
      cb(m_else);
    }

    DEF_MUTABLE_FOR_EACH_SUB_EXPR(IfNode)

    ASTNodeHandle m_pred;
    ASTNodeHandle m_then;
    ASTNodeHandle m_else;
  };

  class LambdaNode : public ASTNode {
  public:
    LambdaNode(const Index_t arg_count, ASTNodeHandle child_expr)
        : ASTNode(ASTNode::Type::Lambda, 0), m_arg_count(arg_count),
          m_child_expr(child_expr) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t &, CB &&cb) {
      cb(m_child_expr);
    }

    DEF_MUTABLE_FOR_EACH_SUB_EXPR(LambdaNode)

    Index_t m_arg_count;
    ASTNodeHandle m_child_expr;
  };

  class CallNode : public ASTNode {
  public:
    CallNode(const Index_t arg_count, ASTNodeHandle target,
             PortableMemPool::HostAccessor_t pool)
        : ASTNode(ASTNode::Type::Call, arg_count + 1), m_target(target),
          m_args(pool[0].template alloc_array<ASTNodeHandle>(arg_count)) {}

    template <typename CB>
    void for_each_sub_expr(PortableMemPool::HostAccessor_t &host_acc, CB &&cb) {
      cb(m_target);
      if (m_args == PortableMemPool::ArrayHandle<ASTNodeHandle>()) {
        return;
      }
      auto *args_data = host_acc[0].deref_handle(m_args);
      for (Index_t i = 0; i < m_args.get_count(); ++i) {
        cb(args_data[i]);
      }
    }

    DEF_MUTABLE_FOR_EACH_SUB_EXPR(CallNode)

    ASTNodeHandle m_target;
    PortableMemPool::ArrayHandle<ASTNodeHandle> m_args;
  };

#undef DEF_MUTABLE_FOR_EACH_SUB_EXPR

  class CompileException {
  public:
    CompileException(const std::string &what) : m_what(what) {}
    const std::string &what() const { return m_what; }

  private:
    std::string m_what;
  };

  Compiler(std::shared_ptr<const SExpr> sexpr,
           cl::sycl::buffer<PortableMemPool> pool)
      : m_sexpr(sexpr), m_mem_pool(pool) {}

  ASTNodeHandle compile() {
    std::list<std::string> initial_bound;
    auto mem_pool_acc =
        m_mem_pool.get_access<cl::sycl::access::mode::read_write>();
    return compile(m_sexpr, initial_bound, mem_pool_acc);
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
                               PortableMemPool::HostAccessor_t mem_pool_acc);
  static ASTNodeHandle
  compile_list_of_sexpr(std::shared_ptr<const SExpr> sexpr,
                        std::list<std::string> bound_identifiers,
                        PortableMemPool::HostAccessor_t mem_pool_acc);

  std::shared_ptr<const SExpr> m_sexpr;
  cl::sycl::buffer<PortableMemPool> m_mem_pool;
};

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(Compiler::ASTNode &node, CB &&cb,
                     UnexpectedCB &&unexpectedCB) {
  switch (node.node_type) {
  case Compiler::ASTNode::Type::Bind:
  case Compiler::ASTNode::Type::BindRec:
    return cb(static_cast<Compiler::BindNode &>(node));
  case Compiler::ASTNode::Type::Call:
    return cb(static_cast<Compiler::CallNode &>(node));
  case Compiler::ASTNode::Type::If:
    return cb(static_cast<Compiler::IfNode &>(node));
  case Compiler::ASTNode::Type::Add:
  case Compiler::ASTNode::Type::Sub:
  case Compiler::ASTNode::Type::Mul:
  case Compiler::ASTNode::Type::Div:
  case Compiler::ASTNode::Type::Equal:
  case Compiler::ASTNode::Type::GreaterThan:
  case Compiler::ASTNode::Type::Remainder:
  case Compiler::ASTNode::Type::Expt:
    return cb(static_cast<Compiler::BinaryOpNode &>(node));
  case Compiler::ASTNode::Type::Floor:
    return cb(static_cast<Compiler::UnaryOpNode &>(node));
  case Compiler::ASTNode::Type::Number:
    return cb(static_cast<Compiler::NumberNode &>(node));
  case Compiler::ASTNode::Type::Identifier:
    return cb(static_cast<Compiler::IdentifierNode &>(node));
  case Compiler::ASTNode::Type::Lambda:
    return cb(static_cast<Compiler::LambdaNode &>(node));
  }
  return unexpectedCB(node);
}

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(const Compiler::ASTNode &node, CB &&cb,
                     UnexpectedCB &&unexpectedCB) {
  return visit(
      const_cast<Compiler::ASTNode &>(node),
      [&](auto &derived) { return cb(std::as_const(derived)); },
      [&](auto &derived) { return unexpectedCB(std::as_const(derived)); });
}
} // namespace FunGPU
