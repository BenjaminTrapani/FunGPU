#include "BlockPrep.hpp"
#include "Compiler.hpp"
#include "PortableMemPool.hpp"
#include "Types.hpp"

namespace FunGPU {
BlockPrep::BlockPrep(const Index_t registersPerBlock,
                     const Index_t instructionsPerCycle,
                     const Index_t cyclesPerBlock,
                     cl::sycl::buffer<PortableMemPool> pool)
    : m_registersPerBlock(registersPerBlock),
      m_instructionsPerCycle(instructionsPerCycle),
      m_cyclesPerBlock(cyclesPerBlock), m_pool(pool) {}

void BlockPrep::GetPrimOps(Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t memPoolAcc,
                           std::vector<Compiler::ASTNodeHandle *> &out) {
  struct PrimOpsExtractor {
    PrimOpsExtractor(Compiler::ASTNodeHandle &inputRoot,
                     PortableMemPool::HostAccessor_t inputMemPoolAcc,
                     std::vector<Compiler::ASTNodeHandle *> &outVec)
        : m_root(inputRoot), m_memPoolAcc(inputMemPoolAcc), m_out(outVec) {}

    bool isLeafNode(const Compiler::ASTNodeHandle handle) {
      const auto &derefdNode = *m_memPoolAcc[0].derefHandle(handle);
      const auto type = derefdNode.m_type;
      switch (type) {
      case Compiler::ASTNode::Type::Number:
      case Compiler::ASTNode::Type::Identifier:
        return true;
      default:
        return false;
      }
    }

    void operator()(Compiler::BindNode &) {
      // TODO pull nested let bindings in top level up, do nothing for now
    }

    void operator()(Compiler::CallNode &call) {
      auto *callArgsData = m_memPoolAcc[0].derefHandle(call.m_args);
      if (isLeafNode(call.m_target)) {
        bool allArgsLeaves = true;
        for (std::size_t i = 0; i < call.m_args.GetCount() && allArgsLeaves;
             ++i) {
          allArgsLeaves = isLeafNode(callArgsData[i]);
        }
        if (allArgsLeaves) {
          m_out.emplace_back(&m_root);
          return;
        }
      }
      GetPrimOps(call.m_target, m_memPoolAcc, m_out);
      for (std::size_t i = 0; i < call.m_args.GetCount(); ++i) {
        GetPrimOps(callArgsData[i], m_memPoolAcc, m_out);
      }
    }

    void operator()(Compiler::BinaryOpNode &binOp) {
      if (isLeafNode(binOp.m_arg0) && isLeafNode(binOp.m_arg1)) {
        m_out.emplace_back(&m_root);
      } else {
        GetPrimOps(binOp.m_arg0, m_memPoolAcc, m_out);
        GetPrimOps(binOp.m_arg1, m_memPoolAcc, m_out);
      }
    }

    void operator()(Compiler::UnaryOpNode &unaryOp) {
      if (isLeafNode(unaryOp.m_arg0)) {
        m_out.emplace_back(&m_root);
      } else {
        GetPrimOps(unaryOp.m_arg0, m_memPoolAcc, m_out);
      }
    }

    void operator()(Compiler::IfNode &ifNode) {
      if (isLeafNode(ifNode.m_pred) && isLeafNode(ifNode.m_then) &&
          isLeafNode(ifNode.m_else)) {
        m_out.emplace_back(&m_root);
      } else {
        GetPrimOps(ifNode.m_pred, m_memPoolAcc, m_out);
      }
    }

    void operator()(const Compiler::NumberNode &) {}

    void operator()(const Compiler::IdentifierNode &) {}

    void operator()(const Compiler::LambdaNode &) {
      m_out.emplace_back(&m_root);
    }

    std::vector<Compiler::ASTNodeHandle *> &m_out;
    PortableMemPool::HostAccessor_t m_memPoolAcc;
    Compiler::ASTNodeHandle &m_root;
  } primOpsExtractor(root, memPoolAcc, out);

  visit(
      *memPoolAcc[0].derefHandle(root), primOpsExtractor,
      [](const auto &node) { throw std::invalid_argument("Unexpected node"); });
}

void BlockPrep::IncreaseBindingRefIndices(
    Compiler::ASTNodeHandle root, const std::size_t increment,
    PortableMemPool::HostAccessor_t memPoolAcc, std::size_t minRefForIncrement,
    const std::set<Compiler::ASTNodeHandle> &identsToExclude) {
  struct BindingIndexIncrementer {
    BindingIndexIncrementer(
        const std::size_t inputIncrement,
        PortableMemPool::HostAccessor_t inputMemPoolAcc,
        const std::size_t minRefForIncrement,
        const std::set<Compiler::ASTNodeHandle> &identsToExclude,
        const Compiler::ASTNodeHandle root)
        : m_increment(inputIncrement), m_memPoolAcc(inputMemPoolAcc),
          m_minRefForIncrement(minRefForIncrement),
          m_toExclude(identsToExclude), m_root(root) {}

    void operator()(const Compiler::BindNode &node) {
      const auto *bindingsData = m_memPoolAcc[0].derefHandle(node.m_bindings);
      const auto isRec = node.m_type == Compiler::ASTNode::Type::BindRec;
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        IncreaseBindingRefIndices(bindingsData[i], m_increment, m_memPoolAcc,
                                  isRec ? m_minRefForIncrement +
                                              node.m_bindings.GetCount()
                                        : m_minRefForIncrement,
                                  m_toExclude);
      }
      IncreaseBindingRefIndices(
          node.m_childExpr, m_increment, m_memPoolAcc,
          m_minRefForIncrement + node.m_bindings.GetCount(), m_toExclude);
    }

    void operator()(const Compiler::CallNode &node) {
      IncreaseBindingRefIndices(node.m_target, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
      const auto *argsData = m_memPoolAcc[0].derefHandle(node.m_args);
      for (size_t i = 0; i < node.m_args.GetCount(); ++i) {
        IncreaseBindingRefIndices(argsData[i], m_increment, m_memPoolAcc,
                                  m_minRefForIncrement, m_toExclude);
      }
    }

    void operator()(const Compiler::IfNode &node) {
      IncreaseBindingRefIndices(node.m_pred, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
      IncreaseBindingRefIndices(node.m_then, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
      IncreaseBindingRefIndices(node.m_else, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
    }

    void operator()(const Compiler::BinaryOpNode &node) {
      IncreaseBindingRefIndices(node.m_arg0, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
      IncreaseBindingRefIndices(node.m_arg1, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
    }

    void operator()(const Compiler::UnaryOpNode &node) {
      IncreaseBindingRefIndices(node.m_arg0, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
    }

    void operator()(const Compiler::NumberNode &) {}

    void operator()(Compiler::IdentifierNode &node) {
      if (node.m_index >= m_minRefForIncrement &&
          m_toExclude.find(m_root) == m_toExclude.end()) {
        node.m_index += m_increment;
      }
    }

    void operator()(Compiler::LambdaNode &node) {
      IncreaseBindingRefIndices(node.m_childExpr, m_increment, m_memPoolAcc,
                                m_minRefForIncrement, m_toExclude);
    }

    const std::size_t m_increment;
    PortableMemPool::HostAccessor_t m_memPoolAcc;
    const std::size_t m_minRefForIncrement;
    const std::set<Compiler::ASTNodeHandle> &m_toExclude;
    Compiler::ASTNodeHandle m_root;
  } incrementer(increment, memPoolAcc, minRefForIncrement, identsToExclude,
                root);

  visit(*memPoolAcc[0].derefHandle(root), incrementer, [](const auto &node) {
    throw std::invalid_argument("Unexpected node");
  });
}

Compiler::ASTNodeHandle
BlockPrep::PrepareForBlockGeneration(Compiler::ASTNodeHandle root) {
  auto memPoolAcc = m_pool.get_access<cl::sycl::access::mode::read_write>();
  return PrepareForBlockGeneration(root, memPoolAcc);
}

void BlockPrep::CollectAllIdentifiers(
    const Compiler::ASTNodeHandle root,
    PortableMemPool::HostAccessor_t memPoolAcc,
    std::set<Compiler::ASTNodeHandle> &result) {
  struct CollectAllHandler {
    CollectAllHandler(const Compiler::ASTNodeHandle root,
                      PortableMemPool::HostAccessor_t memPoolAcc,
                      std::set<Compiler::ASTNodeHandle> &result)
        : m_root(root), m_memPoolAcc(memPoolAcc), m_result(result) {}

    void operator()(const Compiler::BindNode &node) {
      const auto *allBindings = m_memPoolAcc[0].derefHandle(node.m_bindings);
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        CollectAllIdentifiers(allBindings[i], m_memPoolAcc, m_result);
      }
      CollectAllIdentifiers(node.m_childExpr, m_memPoolAcc, m_result);
    }

    void operator()(const Compiler::CallNode &node) {
      const auto *allArgs = m_memPoolAcc[0].derefHandle(node.m_args);
      for (size_t i = 0; i < node.m_args.GetCount(); ++i) {
        CollectAllIdentifiers(allArgs[i], m_memPoolAcc, m_result);
      }
      CollectAllIdentifiers(node.m_target, m_memPoolAcc, m_result);
    }

    void operator()(const Compiler::IfNode &node) {
      CollectAllIdentifiers(node.m_pred, m_memPoolAcc, m_result);
      CollectAllIdentifiers(node.m_then, m_memPoolAcc, m_result);
      CollectAllIdentifiers(node.m_else, m_memPoolAcc, m_result);
    }

    void operator()(const Compiler::BinaryOpNode &node) {
      CollectAllIdentifiers(node.m_arg0, m_memPoolAcc, m_result);
      CollectAllIdentifiers(node.m_arg1, m_memPoolAcc, m_result);
    }

    void operator()(const Compiler::UnaryOpNode &node) {
      CollectAllIdentifiers(node.m_arg0, m_memPoolAcc, m_result);
    }

    void operator()(const Compiler::NumberNode &) {}

    void operator()(const Compiler::IdentifierNode &) {
      m_result.emplace(m_root);
    }

    void operator()(const Compiler::LambdaNode &node) {
      CollectAllIdentifiers(node.m_childExpr, m_memPoolAcc, m_result);
    }

    const Compiler::ASTNodeHandle m_root;
    PortableMemPool::HostAccessor_t m_memPoolAcc;
    std::set<Compiler::ASTNodeHandle> &m_result;
  } collectAllHandler(root, memPoolAcc, result);
  visit(*memPoolAcc[0].derefHandle(root), collectAllHandler,
        [](const auto &unexpected) {
          throw std::invalid_argument("Unexpected node");
        });
}

Compiler::ASTNodeHandle
BlockPrep::RewriteAsPrimOps(Compiler::ASTNodeHandle root,
                            PortableMemPool::HostAccessor_t memPoolAcc) {
  struct RewriteNestedBinaryAsPrimOps {
    RewriteNestedBinaryAsPrimOps(
        const Compiler::ASTNodeHandle inputRoot,
        PortableMemPool::HostAccessor_t inputMemPoolAcc)
        : m_root(inputRoot), m_memPoolAcc(inputMemPoolAcc) {}

    Compiler::ASTNodeHandle operator()(Compiler::BindNode &node) {
      // TODO actually rewrite, pull prim ops up into outer let expression.
      auto *bindingsData = m_memPoolAcc[0].derefHandle(node.m_bindings);
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        bindingsData[i] = RewriteAsPrimOps(bindingsData[i], m_memPoolAcc);
      }
      node.m_childExpr = RewriteAsPrimOps(node.m_childExpr, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(Compiler::CallNode &call) {
      // TODO actually rewrite, pull prim ops up into outer let expression.
      auto *argData = m_memPoolAcc[0].derefHandle(call.m_args);
      for (size_t i = 0; i < call.m_args.GetCount(); ++i) {
        argData[i] = RewriteAsPrimOps(argData[i], m_memPoolAcc);
      }
      call.m_target = RewriteAsPrimOps(call.m_target, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(Compiler::IfNode &ifNode) {
      // TODO pull any prim ops up into new let
      ifNode.m_pred = RewriteAsPrimOps(ifNode.m_pred, m_memPoolAcc);
      ifNode.m_then = RewriteAsPrimOps(ifNode.m_then, m_memPoolAcc);
      ifNode.m_else = RewriteAsPrimOps(ifNode.m_else, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(const Compiler::BinaryOpNode &) {
      std::vector<Compiler::ASTNodeHandle *> primOpsUnderAdd;
      GetPrimOps(m_root, m_memPoolAcc, primOpsUnderAdd);
      // This binary node is a prim op, no need to rewrite this node
      if (primOpsUnderAdd.size() == 1 && primOpsUnderAdd[0] == &m_root) {
        return m_root;
      }
      auto bindNodeForPrimOps =
          m_memPoolAcc[0].template Alloc<Compiler::BindNode>(
              primOpsUnderAdd.size(), false, m_memPoolAcc);
      auto &derefdBindNodeForPrimOps =
          *m_memPoolAcc[0].derefHandle(bindNodeForPrimOps);
      // Only increase references to binding ops if the reference references
      // something above m_root, otherwise should not increment.
      std::set<Compiler::ASTNodeHandle> identsToExclude;
      for (const auto primOpHandle : primOpsUnderAdd) {
        CollectAllIdentifiers(*primOpHandle, m_memPoolAcc, identsToExclude);
      }
      IncreaseBindingRefIndices(m_root, primOpsUnderAdd.size(), m_memPoolAcc, 0,
                                identsToExclude);
      auto *bindingsData =
          m_memPoolAcc[0].derefHandle(derefdBindNodeForPrimOps.m_bindings);
      for (size_t i = 0; i < primOpsUnderAdd.size(); ++i) {
        bindingsData[i] = *primOpsUnderAdd[i];
        *primOpsUnderAdd[i] =
            m_memPoolAcc[0].template Alloc<Compiler::IdentifierNode>(
                primOpsUnderAdd.size() - i - 1);
      }
      derefdBindNodeForPrimOps.m_childExpr =
          RewriteAsPrimOps(m_root, m_memPoolAcc);
      return bindNodeForPrimOps;
    }

    Compiler::ASTNodeHandle operator()(Compiler::UnaryOpNode &node) {
      // TODO maybe rewrite unary op
      node.m_arg0 = RewriteAsPrimOps(node.m_arg0, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(const Compiler::NumberNode &) {
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(const Compiler::IdentifierNode &) {
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(Compiler::LambdaNode &node) {
      node.m_childExpr = RewriteAsPrimOps(node.m_childExpr, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle m_root;
    PortableMemPool::HostAccessor_t m_memPoolAcc;
  } rewriteAsPrimOps(root, memPoolAcc);

  return visit(*memPoolAcc[0].derefHandle(root), rewriteAsPrimOps,
               [&](const auto &node) -> Compiler::ASTNodeHandle {
                 throw std::invalid_argument("Unexpected node");
               });
}

Compiler::ASTNodeHandle BlockPrep::PrepareForBlockGeneration(
    Compiler::ASTNodeHandle root, PortableMemPool::HostAccessor_t memPoolAcc) {
  return RewriteAsPrimOps(root, memPoolAcc);
}
} // namespace FunGPU
