#pragma once

#include "Array.hpp"
#include "PortableMemPool.hpp"
#include "SExpr.hpp"
#include "Types.hpp"
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
    ASTNode(Type type, Index_t frameSize)
        : m_type(type), m_frameSize(frameSize) {}

    Index_t GetFrameCount() const { return m_frameSize; }

    Type m_type;
    Index_t m_frameSize;
  };

  using ASTNodeHandle = PortableMemPool::Handle<ASTNode>;

  class BindNode : public ASTNode {
  public:
    BindNode(const Index_t numBindings, const bool isRec,
             PortableMemPool::HostAccessor_t pool)
        : ASTNode(isRec ? Type::BindRec : Type::Bind, numBindings),
          m_bindings(pool[0].template AllocArray<ASTNodeHandle>(numBindings)) {}
    PortableMemPool::ArrayHandle<ASTNodeHandle> m_bindings;
    ASTNodeHandle m_childExpr;
  };

  class UnaryOpNode : public ASTNode {
  public:
    UnaryOpNode(ASTNode::Type type, ASTNodeHandle arg0)
        : ASTNode(type, 1), m_arg0(arg0) {}

    ASTNodeHandle m_arg0;
  };

  class BinaryOpNode : public ASTNode {
  public:
    BinaryOpNode(ASTNode::Type type, ASTNodeHandle arg0, ASTNodeHandle arg1)
        : ASTNode(type, 2), m_arg0(arg0), m_arg1(arg1) {}

    ASTNodeHandle m_arg0;
    ASTNodeHandle m_arg1;
  };

  class NumberNode : public ASTNode {
  public:
    NumberNode(const Float_t value)
        : ASTNode(ASTNode::Type::Number, 0), m_value(value) {}
    Float_t m_value;
  };

  class IdentifierNode : public ASTNode {
  public:
    IdentifierNode(const Index_t index)
        : ASTNode(ASTNode::Type::Identifier, 0), m_index(index) {}
    Index_t m_index;
  };

  class IfNode : public ASTNode {
  public:
    IfNode(ASTNodeHandle pred, ASTNodeHandle then, ASTNodeHandle elseExpr)
        : ASTNode(ASTNode::Type::If, 1), m_pred(pred), m_then(then),
          m_else(elseExpr) {}
    ASTNodeHandle m_pred;
    ASTNodeHandle m_then;
    ASTNodeHandle m_else;
  };

  class LambdaNode : public ASTNode {
  public:
    LambdaNode(const Index_t argCount, ASTNodeHandle childExpr)
        : ASTNode(ASTNode::Type::Lambda, 0), m_argCount(argCount),
          m_childExpr(childExpr) {}
    Index_t m_argCount;
    ASTNodeHandle m_childExpr;
  };

  class CallNode : public ASTNode {
  public:
    CallNode(const Index_t argCount, ASTNodeHandle target,
             PortableMemPool::HostAccessor_t pool)
        : ASTNode(ASTNode::Type::Call, argCount + 1), m_target(target),
          m_args(pool[0].template AllocArray<ASTNodeHandle>(argCount)) {}
    ASTNodeHandle m_target;
    PortableMemPool::ArrayHandle<ASTNodeHandle> m_args;
  };

  class CompileException {
  public:
    CompileException(const std::string &what) : m_what(what) {}
    const std::string &What() const { return m_what; }

  private:
    std::string m_what;
  };

  Compiler(std::shared_ptr<const SExpr> sexpr,
           cl::sycl::buffer<PortableMemPool> pool)
      : m_sExpr(sexpr), m_memPool(pool) {}

  ASTNodeHandle Compile() {
    std::list<std::string> initialBound;
    auto memPoolAcc =
        m_memPool.get_access<cl::sycl::access::mode::read_write>();
    return Compile(m_sExpr, initialBound, memPoolAcc);
  }

  void DebugPrintAST(ASTNodeHandle rootOfAST);
  void DeallocateAST(const ASTNodeHandle rootOfAST);

private:
  static void DebugPrintAST(ASTNodeHandle rootOfAST,
                            PortableMemPool::HostAccessor_t memPoolAcc);
  static void DeallocateAST(const ASTNodeHandle &handle,
                            PortableMemPool::HostAccessor_t memPoolAcc);

  static ASTNodeHandle Compile(std::shared_ptr<const SExpr> sexpr,
                               std::list<std::string> boundIdentifiers,
                               PortableMemPool::HostAccessor_t memPoolAcc);
  static ASTNodeHandle
  CompileListOfSExpr(std::shared_ptr<const SExpr> sexpr,
                     std::list<std::string> boundIdentifiers,
                     PortableMemPool::HostAccessor_t memPoolAcc);

  std::shared_ptr<const SExpr> m_sExpr;
  cl::sycl::buffer<PortableMemPool> m_memPool;
};

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(const Compiler::ASTNode &node, CB &&cb,
                     UnexpectedCB &&unexpectedCB) {
  switch (node.m_type) {
  case Compiler::ASTNode::Type::Bind:
  case Compiler::ASTNode::Type::BindRec:
    return cb(static_cast<const Compiler::BindNode &>(node));
  case Compiler::ASTNode::Type::Call:
    return cb(static_cast<const Compiler::CallNode &>(node));
  case Compiler::ASTNode::Type::If:
    return cb(static_cast<const Compiler::IfNode &>(node));
  case Compiler::ASTNode::Type::Add:
  case Compiler::ASTNode::Type::Sub:
  case Compiler::ASTNode::Type::Mul:
  case Compiler::ASTNode::Type::Div:
  case Compiler::ASTNode::Type::Equal:
  case Compiler::ASTNode::Type::GreaterThan:
  case Compiler::ASTNode::Type::Remainder:
  case Compiler::ASTNode::Type::Expt:
    return cb(static_cast<const Compiler::BinaryOpNode &>(node));
  case Compiler::ASTNode::Type::Floor:
    return cb(static_cast<const Compiler::UnaryOpNode &>(node));
  case Compiler::ASTNode::Type::Number:
    return cb(static_cast<const Compiler::NumberNode &>(node));
  case Compiler::ASTNode::Type::Identifier:
    return cb(static_cast<const Compiler::IdentifierNode &>(node));
  case Compiler::ASTNode::Type::Lambda:
    return cb(static_cast<const Compiler::LambdaNode &>(node));
  }
  return unexpectedCB(node);
}

template <typename CB, typename UnexpectedCB>
decltype(auto) visit(Compiler::ASTNode &node, CB &&cb,
                     UnexpectedCB &&unexpectedCB) {
  return visit(
      static_cast<const Compiler::ASTNode &>(node),
      [&](const auto &derived) {
        return cb(const_cast<std::decay_t<decltype(derived)> &>(derived));
      },
      [&](const auto &derived) {
        return unexpectedCB(
            const_cast<std::decay_t<decltype(derived)> &>(derived));
      });
}
} // namespace FunGPU
