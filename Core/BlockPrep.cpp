#include "BlockPrep.hpp"
#include "Compiler.hpp"
#include "PortableMemPool.hpp"
#include "Types.hpp"
#include "Visitor.hpp"
#include <map>

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
      if (isLeafNode(ifNode.m_pred)) {
        m_out.emplace_back(&m_root);
      } else {
        GetPrimOps(ifNode.m_pred, m_memPoolAcc, m_out);
      }
    }

    void operator()(const Compiler::NumberNode &) {
      m_out.emplace_back(&m_root);
    }

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
  visit(
      *memPoolAcc[0].derefHandle(root),
      Visitor{
          [&](const Compiler::BindNode &node) {
            const auto *bindingsData =
                memPoolAcc[0].derefHandle(node.m_bindings);
            const auto isRec = node.m_type == Compiler::ASTNode::Type::BindRec;
            for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
              IncreaseBindingRefIndices(bindingsData[i], increment, memPoolAcc,
                                        isRec ? minRefForIncrement +
                                                    node.m_bindings.GetCount()
                                              : minRefForIncrement,
                                        identsToExclude);
            }
            IncreaseBindingRefIndices(node.m_childExpr, increment, memPoolAcc,
                                      minRefForIncrement +
                                          node.m_bindings.GetCount(),
                                      identsToExclude);
          },
          [&](Compiler::IdentifierNode &node) {
            if (node.m_index >= minRefForIncrement &&
                identsToExclude.find(root) == identsToExclude.end()) {
              node.m_index += increment;
            }
          },
          [&](const Compiler::LambdaNode &node) {
            IncreaseBindingRefIndices(node.m_childExpr, increment, memPoolAcc,
                                      minRefForIncrement + node.m_argCount,
                                      identsToExclude);
          },
          [&](const auto &standard_node) {
            standard_node.for_each_sub_expr(memPoolAcc, [&](const auto handle) {
              IncreaseBindingRefIndices(handle, increment, memPoolAcc,
                                        minRefForIncrement, identsToExclude);
            });
          }},
      [](const auto &node) { throw std::invalid_argument("Unexpected node"); });
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
  visit(
      *memPoolAcc[0].derefHandle(root),
      [&](auto &node) {
        if (types.find(node.m_type) != types.end()) {
          result.emplace(&root);
          return;
        }
        node.for_each_sub_expr(memPoolAcc, [&](auto &child_expr) {
          BlockPrep::CollectAll(child_expr, memPoolAcc, types, result);
        });
      },
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
      return RewriteAsPrimOps(bindNodeForPrimOps, m_memPoolAcc);
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
        std::set<Compiler::ASTNodeHandle *> curBindings;
        CollectAll(bindingsData[i], m_memPoolAcc,
                   {Compiler::ASTNode::Type::Bind,
                    Compiler::ASTNode::Type::BindRec,
                    Compiler::ASTNode::Type::Lambda},
                   curBindings);
        // Collect lambdas to prevent collection of binds inside lambda, cannot
        // move these.
        for (auto elem : curBindings) {
          const auto type = m_memPoolAcc[0].derefHandle(*elem)->m_type;
          switch (type) {
          case Compiler::ASTNode::Type::Bind:
          case Compiler::ASTNode::Type::BindRec:
            allNestedBindings.emplace(elem);
            break;
          case Compiler::ASTNode::Type::Lambda:
            *elem = RewriteAsPrimOps(*elem, m_memPoolAcc);
            break;
          default:
            throw std::invalid_argument("Unexpected node type");
          }
        }
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
      std::vector<Compiler::ASTNodeHandle *> primOpsUnderPred;
      GetPrimOps(m_root, m_memPoolAcc, primOpsUnderPred);
      // Branch is already a prim op. Rewrite both sub branches.
      if (primOpsUnderPred.size() == 1 && primOpsUnderPred[0] == &m_root) {
        ifNode.m_then = RewriteAsPrimOps(ifNode.m_then, m_memPoolAcc);
        ifNode.m_else = RewriteAsPrimOps(ifNode.m_else, m_memPoolAcc);
        return m_root;
      }

      return PullPrimOpsAbove(m_root, primOpsUnderPred);
    }

    Compiler::ASTNodeHandle operator()(const Compiler::BinaryOpNode &) {
      return pullPrimOpsUnderUp(m_root);
    }

    Compiler::ASTNodeHandle operator()(Compiler::UnaryOpNode &node) {
      return pullPrimOpsUnderUp(m_root);
    }

    Compiler::ASTNodeHandle operator()(const Compiler::NumberNode &) {
      const auto bind_node =
          m_memPoolAcc[0].Alloc<Compiler::BindNode>(1, false, m_memPoolAcc);
      auto &derefd_bind_node = *m_memPoolAcc[0].derefHandle(bind_node);
      auto *binding_data =
          m_memPoolAcc[0].derefHandle(derefd_bind_node.m_bindings);
      binding_data[0] = m_root;
      derefd_bind_node.m_childExpr =
          m_memPoolAcc[0].Alloc<Compiler::IdentifierNode>(0);
      return bind_node;
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

/*
Compiler::ASTNodeHandle
BlockPrep::replace_numeric_constants_with_idents_per_lambda(Compiler::ASTNodeHandle
root, PortableMemPool::HostAccessor_t& mem_pool_acc) {
  visit(*mem_pool_acc[0].derefHandle(root), Visitor{ [&](const
Compiler::LambdaNode& node) { std::set<Compiler::ASTNodeHandle *>
numeric_constants_under_lambda; CollectAll(node.m_childExpr, mem_pool_acc,
                {Compiler::ASTNode::Type::Number,
Compiler::ASTNode::Type::Lambda, Compiler::ASTNode::Type::Bind,
                  Compiler::ASTNode::Type::BindRec},
numeric_constants_under_lambda); for (auto iter =
numeric_constants_under_lambda.begin(); iter !=
numeric_constants_under_lambda.end();) { const auto& maybe_numeric_node =
*mem_pool_acc[0].derefHandle(**iter); if (maybe_numeric_node.m_type !=
Compiler::ASTNode::Type::Number) { iter =
numeric_constants_under_lambda.erase(iter); } else {
        ++iter;
      }
    }
    std::map<Float_t, std::set<Compiler::ASTNodeHandle *>>
unique_constants_to_refs; for (auto* node_handle :
numeric_constants_under_lambda) { const auto& numeric_node = static_cast<const
Compiler::NumberNode&>(*mem_pool_acc[0].derefHandle(*node_handle));
      unique_constants_to_refs[numeric_node.m_value].emplace(node_handle);
    }
    if (unique_constants_to_refs.empty()) {
      node.m_childExpr =
replace_numeric_constants_with_idents_per_lambda(node.m_childExpr,
mem_pool_acc); return root;
    }
    IncreaseBindingRefIndices(root, unique_constants_to_refs.size(),
mem_pool_acc, 0,
                                {});
    auto bind_node_for_constants =
          mem_pool_acc[0].template Alloc<Compiler::BindNode>(
              unique_constants_to_refs.size(), false, mem_pool_acc);
    auto &derefd_bind_node_for_constants =
        *mem_pool_acc[0].derefHandle(bind_node_for_constants);
    auto* binding_data =
mem_pool_acc[0].derefHandle(derefd_bind_node_for_constants.m_bindings); Index_t
i = 0; for (auto& [float_val, node_handles] : unique_constants_to_refs) { if
(node_handles.empty()) { throw std::invalid_argument("Should have at least one
element");
      }
      const auto& first_node_handle = **node_handles.begin();
      binding_data[i] = first_node_handle;
      for (auto* node_handle : node_handles) {
        *node_handle = mem_pool_acc[0].template
Alloc<Compiler::IdentifierNode>(unique_constants_to_refs.size() - i - 1);
      }
      ++i;
    }
    derefd_bind_node_for_constants.m_childExpr =
replace_numeric_constants_with_idents_per_lambda(root, mem_pool_acc); return
derefd_bind_node_for_constants;
  },
  [&](const auto& node) {
    node.for_each_sub_expr(mem_pool_acc, [&](const auto& sub_expr_handle) {
      return replace_numeric_constants_with_idents_per_lambda(sub_expr_handle,
mem_pool_acc);
    });
    return root;
  }}, [](const auto& unexpected_node) {
    throw std::invalid_argument("Unexpected node");
  });
}
*/

Compiler::ASTNodeHandle BlockPrep::wrap_in_no_arg_lambda(
    Compiler::ASTNodeHandle root,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  const auto outer_lambda =
      mem_pool_acc[0].Alloc<Compiler::LambdaNode>(0, root);
  const auto call_expr =
      mem_pool_acc[0].Alloc<Compiler::CallNode>(0, outer_lambda, mem_pool_acc);
  return call_expr;
}

Compiler::ASTNodeHandle BlockPrep::PrepareForBlockGeneration(
    Compiler::ASTNodeHandle root, PortableMemPool::HostAccessor_t memPoolAcc) {
  const auto wrapped_in_no_arg_lambda = wrap_in_no_arg_lambda(root, memPoolAcc);
  const auto rewritten_as_prim_ops =
      RewriteAsPrimOps(wrapped_in_no_arg_lambda, memPoolAcc);
  return rewritten_as_prim_ops;
}
} // namespace FunGPU
