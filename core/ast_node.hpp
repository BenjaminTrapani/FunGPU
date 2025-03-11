#pragma once

#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"

namespace FunGPU {
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
        m_bindings(pool[0].template alloc_array<ASTNodeHandle>(num_bindings)) {}

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

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(ASTNode &node, CB &&cb, UnexpectedCB &&unexpectedCB) {
  switch (node.node_type) {
  case ASTNode::Type::Bind:
  case ASTNode::Type::BindRec:
    return cb(static_cast<BindNode &>(node));
  case ASTNode::Type::Call:
    return cb(static_cast<CallNode &>(node));
  case ASTNode::Type::If:
    return cb(static_cast<IfNode &>(node));
  case ASTNode::Type::Add:
  case ASTNode::Type::Sub:
  case ASTNode::Type::Mul:
  case ASTNode::Type::Div:
  case ASTNode::Type::Equal:
  case ASTNode::Type::GreaterThan:
  case ASTNode::Type::Remainder:
  case ASTNode::Type::Expt:
    return cb(static_cast<BinaryOpNode &>(node));
  case ASTNode::Type::Floor:
    return cb(static_cast<UnaryOpNode &>(node));
  case ASTNode::Type::Number:
    return cb(static_cast<NumberNode &>(node));
  case ASTNode::Type::Identifier:
    return cb(static_cast<IdentifierNode &>(node));
  case ASTNode::Type::Lambda:
    return cb(static_cast<LambdaNode &>(node));
  }
  return unexpectedCB(node);
}

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(const ASTNode &node, CB &&cb,
                     UnexpectedCB &&unexpectedCB) {
  return visit(
      const_cast<ASTNode &>(node),
      [&](auto &derived) { return cb(std::as_const(derived)); },
      [&](auto &derived) { return unexpectedCB(std::as_const(derived)); });
}
} // namespace FunGPU
