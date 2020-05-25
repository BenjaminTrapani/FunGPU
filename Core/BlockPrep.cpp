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

    void operator()(Compiler::BindNode &node) {
      // For recursive bindings, the bound expressions are already in new
      // binding space, cannot pull up farther
      if (node.m_type == Compiler::ASTNode::Type::BindRec) {
        m_out.emplace_back(&m_root);
        return;
      }
      auto *bindingsData = m_memPoolAcc[0].derefHandle(node.m_bindings);
      bool allBindingsLeaves = false;
      for (size_t i = 0; i < node.m_bindings.GetCount() && allBindingsLeaves;
           ++i) {
        allBindingsLeaves = isLeafNode(bindingsData[i]);
      }
      if (allBindingsLeaves) {
        m_out.emplace_back(&m_root);
        return;
      }
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        GetPrimOps(bindingsData[i], m_memPoolAcc, m_out);
      }
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
    const Compiler::ASTNodeHandle root, const std::size_t increment,
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
                                m_minRefForIncrement + node.m_argCount,
                                m_toExclude);
    }

    const std::size_t m_increment;
    PortableMemPool::HostAccessor_t m_memPoolAcc;
    const std::size_t m_minRefForIncrement;
    const std::set<Compiler::ASTNodeHandle> &m_toExclude;
    const Compiler::ASTNodeHandle m_root;
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

void BlockPrep::CollectAll(Compiler::ASTNodeHandle &root,
                           PortableMemPool::HostAccessor_t memPoolAcc,
                           const std::set<Compiler::ASTNode::Type> &types,
                           std::set<Compiler::ASTNodeHandle *> &result) {
  struct CollectAllHandler {
    CollectAllHandler(Compiler::ASTNodeHandle &root,
                      PortableMemPool::HostAccessor_t memPoolAcc,
                      const std::set<Compiler::ASTNode::Type> &types,
                      std::set<Compiler::ASTNodeHandle *> &result)
        : m_root(root), m_memPoolAcc(memPoolAcc), m_types(types),
          m_result(result) {}

    void operator()(Compiler::BindNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
        return;
      }
      auto *allBindings = m_memPoolAcc[0].derefHandle(node.m_bindings);
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        CollectAll(allBindings[i], m_memPoolAcc, m_types, m_result);
      }
      CollectAll(node.m_childExpr, m_memPoolAcc, m_types, m_result);
    }

    void operator()(Compiler::CallNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
        return;
      }
      auto *allArgs = m_memPoolAcc[0].derefHandle(node.m_args);
      for (size_t i = 0; i < node.m_args.GetCount(); ++i) {
        CollectAll(allArgs[i], m_memPoolAcc, m_types, m_result);
      }
      CollectAll(node.m_target, m_memPoolAcc, m_types, m_result);
    }

    void operator()(Compiler::IfNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
        return;
      }
      CollectAll(node.m_pred, m_memPoolAcc, m_types, m_result);
      CollectAll(node.m_then, m_memPoolAcc, m_types, m_result);
      CollectAll(node.m_else, m_memPoolAcc, m_types, m_result);
    }

    void operator()(Compiler::BinaryOpNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
        return;
      }
      CollectAll(node.m_arg0, m_memPoolAcc, m_types, m_result);
      CollectAll(node.m_arg1, m_memPoolAcc, m_types, m_result);
    }

    void operator()(Compiler::UnaryOpNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
        return;
      }
      CollectAll(node.m_arg0, m_memPoolAcc, m_types, m_result);
    }

    void operator()(Compiler::NumberNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
      }
    }

    void operator()(Compiler::IdentifierNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
      }
    }

    void operator()(Compiler::LambdaNode &node) {
      if (m_types.find(node.m_type) != m_types.end()) {
        m_result.emplace(&m_root);
        return;
      }
      CollectAll(node.m_childExpr, m_memPoolAcc, m_types, m_result);
    }

    Compiler::ASTNodeHandle &m_root;
    PortableMemPool::HostAccessor_t m_memPoolAcc;
    const std::set<Compiler::ASTNode::Type> &m_types;
    std::set<Compiler::ASTNodeHandle *> &m_result;
  } collectAllHandler(root, memPoolAcc, types, result);
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

    std::set<Compiler::ASTNodeHandle>
    toConcrete(const std::set<Compiler::ASTNodeHandle *> &original) {
      std::set<Compiler::ASTNodeHandle> result;
      for (auto val : original) {
        result.emplace(*val);
      }
      return result;
    }

    Compiler::ASTNodeHandle
    PullPrimOpsAbove(Compiler::ASTNodeHandle &root,
                     std::vector<Compiler::ASTNodeHandle *> &primOps) {
      auto bindNodeForPrimOps =
          m_memPoolAcc[0].template Alloc<Compiler::BindNode>(
              primOps.size(), false, m_memPoolAcc);
      auto &derefdBindNodeForPrimOps =
          *m_memPoolAcc[0].derefHandle(bindNodeForPrimOps);
      // Only increase references to binding ops if the reference references
      // something above root, otherwise should not increment.
      std::set<Compiler::ASTNodeHandle *> identsToExclude;
      for (auto primOpHandle : primOps) {
        CollectAll(*primOpHandle, m_memPoolAcc,
                   {Compiler::ASTNode::Type::Identifier}, identsToExclude);
      }
      IncreaseBindingRefIndices(root, primOps.size(), m_memPoolAcc, 0,
                                toConcrete(identsToExclude));
      auto *bindingsData =
          m_memPoolAcc[0].derefHandle(derefdBindNodeForPrimOps.m_bindings);
      for (size_t i = 0; i < primOps.size(); ++i) {
        bindingsData[i] = *primOps[i];
        *primOps[i] = m_memPoolAcc[0].template Alloc<Compiler::IdentifierNode>(
            primOps.size() - i - 1);
      }
      derefdBindNodeForPrimOps.m_childExpr =
          RewriteAsPrimOps(root, m_memPoolAcc);
      return bindNodeForPrimOps;
    }

    Compiler::ASTNodeHandle pullPrimOpsUnderUp(Compiler::ASTNodeHandle &root) {
      std::vector<Compiler::ASTNodeHandle *> primOpsUnderAdd;
      GetPrimOps(root, m_memPoolAcc, primOpsUnderAdd);
      // This node is a prim op, no need to rewrite this node
      if (primOpsUnderAdd.size() == 1 && primOpsUnderAdd[0] == &root) {
        return root;
      }
      return PullPrimOpsAbove(root, primOpsUnderAdd);
    }

    Compiler::ASTNodeHandle operator()(Compiler::BindNode &node) {
      // TODO handle recursive bindings correctly. Not much to do so far as
      // rewriting these. Probably best to rewrite each sub-tree as usual and
      // leave recursive bindings as before, generating one block of
      // instructions for each bound expression in the recursive binding. This
      // is actually required do enable correct indirect calls in evaluation.
      if (node.m_type == Compiler::ASTNode::Type::BindRec) {
        node.m_childExpr = RewriteAsPrimOps(node.m_childExpr, m_memPoolAcc);
        auto *bindingData = m_memPoolAcc[0].derefHandle(node.m_bindings);
        for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
          bindingData[i] = RewriteAsPrimOps(bindingData[i], m_memPoolAcc);
        }
        return m_root;
      }

      std::set<Compiler::ASTNodeHandle *> allNestedBindings;
      auto *bindingsData = m_memPoolAcc[0].derefHandle(node.m_bindings);
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        CollectAll(
            bindingsData[i], m_memPoolAcc,
            {Compiler::ASTNode::Type::Bind, Compiler::ASTNode::Type::BindRec},
            allNestedBindings);
      }

      if (!allNestedBindings.empty()) {
        // Move top level of nested let expressions in bound expressions to
        // chain of lets above the current binding expression. Order does not
        // matter, all in the same scope.
        Compiler::ASTNodeHandle mostRecentUnestedLet;
        const auto outermostGeneratedLet = **allNestedBindings.begin();
        for (auto nestedBindHandle : allNestedBindings) {
          std::set<Compiler::ASTNodeHandle *> identsToExclude;
          CollectAll(*nestedBindHandle, m_memPoolAcc,
                     {Compiler::ASTNode::Type::Identifier}, identsToExclude);
          auto &nestedBindExpr = static_cast<Compiler::BindNode &>(
              *m_memPoolAcc[0].derefHandle(*nestedBindHandle));
          IncreaseBindingRefIndices(
              m_root, nestedBindExpr.m_bindings.GetCount(), m_memPoolAcc, 0,
              toConcrete(identsToExclude));
          if (mostRecentUnestedLet != Compiler::ASTNodeHandle()) {
            auto &derefdMostRecentUnested = static_cast<Compiler::BindNode &>(
                *m_memPoolAcc[0].derefHandle(mostRecentUnestedLet));
            derefdMostRecentUnested.m_childExpr = *nestedBindHandle;
          }
          const auto tmpNestedBindHandle = *nestedBindHandle;
          *nestedBindHandle = nestedBindExpr.m_childExpr;
          nestedBindExpr.m_childExpr = m_root;
          mostRecentUnestedLet = tmpNestedBindHandle;
        }
        return RewriteAsPrimOps(outermostGeneratedLet, m_memPoolAcc);
      }

      // Decompose complex bound expressions in current scope into their own let
      // expressions.
      std::vector<Compiler::ASTNodeHandle *> primOpsForBindings;
      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        std::vector<Compiler::ASTNodeHandle *> primOpsHere;
        GetPrimOps(bindingsData[i], m_memPoolAcc, primOpsHere);
        if (primOpsHere.size() > 1 ||
            (primOpsHere.size() == 1 && primOpsHere[0] != &bindingsData[i])) {
          primOpsForBindings.insert(primOpsForBindings.end(),
                                    primOpsHere.begin(), primOpsHere.end());
        }
      }
      if (!primOpsForBindings.empty()) {
        return PullPrimOpsAbove(m_root, primOpsForBindings);
      }
      // Let expression is completely decomposed, decompose the child expression
      // and return the root.
      node.m_childExpr = RewriteAsPrimOps(node.m_childExpr, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(Compiler::CallNode &call) {
      return pullPrimOpsUnderUp(m_root);
    }

    Compiler::ASTNodeHandle operator()(Compiler::IfNode &ifNode) {
      // TODO fix this, need to do some custom logic to pull primitives only on
      // pred up.
      ifNode.m_pred = pullPrimOpsUnderUp(ifNode.m_pred);
      // Need to leave primitives in branches where they are (at least under the
      // if)
      ifNode.m_then = RewriteAsPrimOps(ifNode.m_then, m_memPoolAcc);
      ifNode.m_else = RewriteAsPrimOps(ifNode.m_else, m_memPoolAcc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(const Compiler::BinaryOpNode &) {
      return pullPrimOpsUnderUp(m_root);
    }

    Compiler::ASTNodeHandle operator()(Compiler::UnaryOpNode &node) {
      return pullPrimOpsUnderUp(m_root);
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
