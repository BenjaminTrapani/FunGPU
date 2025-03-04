#include "Core/BlockPrep.hpp"
#include "Core/CollectAllASTNodes.hpp"
#include "Core/Visitor.hpp"

namespace FunGPU {
namespace {
template <typename ContainedType>
std::set<std::remove_cvref_t<ContainedType>>
toConcrete(const std::set<ContainedType *> &original) {
  std::set<std::remove_cvref_t<ContainedType>> result;
  for (const auto val : original) {
    result.emplace(*val);
  }
  return result;
}
} // namespace

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

Compiler::ASTNodeHandle
BlockPrep::RewriteAsPrimOps(Compiler::ASTNodeHandle root,
                            PortableMemPool::HostAccessor_t memPoolAcc) {
  struct RewriteNestedBinaryAsPrimOps {
    RewriteNestedBinaryAsPrimOps(
        const Compiler::ASTNodeHandle inputRoot,
        PortableMemPool::HostAccessor_t inputMemPoolAcc)
        : m_root(inputRoot), m_memPoolAcc(inputMemPoolAcc) {}

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
        CollectAllASTNodes(*primOpHandle, m_memPoolAcc,
                           {Compiler::ASTNode::Type::Identifier},
                           identsToExclude);
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
      if (node.m_type != Compiler::ASTNode::Type::Bind) {
        throw std::invalid_argument("Unexpected binding node type");
      }

      std::set<Compiler::ASTNodeHandle *> allNestedBindings;
      auto *bindingsData = m_memPoolAcc[0].derefHandle(node.m_bindings);
      std::set<Compiler::ASTNodeHandle *> boundBranches;

      for (size_t i = 0; i < node.m_bindings.GetCount(); ++i) {
        std::set<Compiler::ASTNodeHandle *> curBindings;
        CollectAllASTNodes(
            bindingsData[i], m_memPoolAcc,
            {Compiler::ASTNode::Type::Bind, Compiler::ASTNode::Type::BindRec,
             Compiler::ASTNode::Type::Lambda, Compiler::ASTNode::Type::If},
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
          case Compiler::ASTNode::Type::If:
            boundBranches.emplace(elem);
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
          CollectAllASTNodes(*nestedBindHandle, m_memPoolAcc,
                             {Compiler::ASTNode::Type::Identifier},
                             identsToExclude);
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

      if (!boundBranches.empty()) {
        // Reformat (let ((x 1) (bound-result (if a b c)) (y 2)) body-expr) as
        //          (let ((x 1) (body_cont (lambda (bound-result) (let ((y 2))
        //          body-expr)))))
        //                (if a (body-cont b) (body-cont c)))
        // Locate the position of the first bound branch.
        // Split the bind expression around the branch.
        // Increase binding indices for all identifiers in bindings to the right
        // hand side of the split by the number of identifiers on the lhs of the
        // split Create a new binding for the continuation of the body
        // expression. Make the right hand side of the split bindings the child
        // expression of the continuation binding lambda. The expression in the
        // tail position is the original branch that invokes the continuation.
        const auto &first_branch_handle = **boundBranches.begin();
        const auto position_of_first_branch_handle =
            &first_branch_handle - bindingsData;
        const auto bind_split_lhs = m_memPoolAcc[0].Alloc<Compiler::BindNode>(
            position_of_first_branch_handle, false, m_memPoolAcc);
        auto &bind_split_lhs_ref = *m_memPoolAcc[0].derefHandle(bind_split_lhs);
        auto *bind_split_lhs_bindings_data =
            m_memPoolAcc[0].derefHandle(bind_split_lhs_ref.m_bindings);
        std::copy(bindingsData, bindingsData + position_of_first_branch_handle,
                  bind_split_lhs_bindings_data);
        const auto bind_split_rhs = m_memPoolAcc[0].Alloc<Compiler::BindNode>(
            node.m_bindings.GetCount() - position_of_first_branch_handle - 1,
            false, m_memPoolAcc);
        auto &bind_split_rhs_ref = *m_memPoolAcc[0].derefHandle(bind_split_rhs);
        bind_split_rhs_ref.m_childExpr = node.m_childExpr;
        auto *bind_split_rhs_bindings_data =
            m_memPoolAcc[0].derefHandle(bind_split_rhs_ref.m_bindings);
        std::copy(bindingsData + position_of_first_branch_handle + 1,
                  bindingsData + node.m_bindings.GetCount(),
                  bind_split_rhs_bindings_data);
        for (size_t i = 0; i < bind_split_rhs_ref.m_bindings.GetCount(); ++i) {
          IncreaseBindingRefIndices(bind_split_rhs_bindings_data[i],
                                    bind_split_lhs_ref.m_bindings.GetCount() +
                                        1,
                                    m_memPoolAcc, 0, {});
        }
        IncreaseBindingRefIndices(first_branch_handle,
                                  bind_split_lhs_ref.m_bindings.GetCount() + 1,
                                  m_memPoolAcc, 0, {});

        const auto continuation_lambda =
            m_memPoolAcc[0].Alloc<Compiler::LambdaNode>(1, bind_split_rhs);
        const auto bind_node_for_continuation =
            m_memPoolAcc[0].Alloc<Compiler::BindNode>(1, false, m_memPoolAcc);
        auto &bind_node_for_continuation_ref =
            *m_memPoolAcc[0].derefHandle(bind_node_for_continuation);
        auto *bindings_data = m_memPoolAcc[0].derefHandle(
            bind_node_for_continuation_ref.m_bindings);
        bindings_data[0] = continuation_lambda;
        bind_split_lhs_ref.m_childExpr = bind_node_for_continuation;
        bind_node_for_continuation_ref.m_childExpr = first_branch_handle;

        auto &first_branch_data = static_cast<Compiler::IfNode &>(
            *m_memPoolAcc[0].derefHandle(first_branch_handle));
        auto call_body_cont_with_true_branch =
            m_memPoolAcc[0].Alloc<Compiler::CallNode>(
                1, m_memPoolAcc[0].Alloc<Compiler::IdentifierNode>(0),
                m_memPoolAcc);
        auto &call_true_data = static_cast<Compiler::CallNode &>(
            *m_memPoolAcc[0].derefHandle(call_body_cont_with_true_branch));
        m_memPoolAcc[0].derefHandle(call_true_data.m_args)[0] =
            first_branch_data.m_then;
        auto call_body_cont_with_false_branch =
            m_memPoolAcc[0].Alloc<Compiler::CallNode>(
                1, m_memPoolAcc[0].Alloc<Compiler::IdentifierNode>(0),
                m_memPoolAcc);
        auto &call_false_data = static_cast<Compiler::CallNode &>(
            *m_memPoolAcc[0].derefHandle(call_body_cont_with_false_branch));
        m_memPoolAcc[0].derefHandle(call_false_data.m_args)[0] =
            first_branch_data.m_else;
        first_branch_data.m_then = call_body_cont_with_true_branch;
        first_branch_data.m_else = call_body_cont_with_false_branch;
        m_memPoolAcc[0].DeallocArray(node.m_bindings);
        m_memPoolAcc[0].Dealloc(m_root);
        return RewriteAsPrimOps(bind_split_lhs, m_memPoolAcc);
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
        const auto original_root = m_root;
        const auto maybe_rewrite_branch_as_call =
            [&](Compiler::ASTNodeHandle &branch) -> Compiler::ASTNodeHandle {
          // Pred and then can be anything except structures influencing control
          // flow / block layout.
          const auto &branch_node = *m_memPoolAcc[0].derefHandle(branch);
          switch (branch_node.m_type) {
          case Compiler::ASTNode::Type::Bind:
          case Compiler::ASTNode::Type::BindRec:
          case Compiler::ASTNode::Type::If: {
            // Create no arg lambda containing this block. Pull it above, and
            // generate call to it here.
            auto bind_node_for_lambda =
                m_memPoolAcc[0].template Alloc<Compiler::BindNode>(
                    1, false, m_memPoolAcc);
            auto &derefd_bind_node =
                *m_memPoolAcc[0].derefHandle(bind_node_for_lambda);
            // Only increase references to binding ops if the reference
            // references something above root, otherwise should not increment.
            std::set<Compiler::ASTNodeHandle *> identsToExclude;
            CollectAllASTNodes(branch, m_memPoolAcc,
                               {Compiler::ASTNode::Type::Identifier},
                               identsToExclude);
            IncreaseBindingRefIndices(original_root, 1, m_memPoolAcc, 0,
                                      toConcrete(identsToExclude));
            auto no_arg_lambda = static_cast<Compiler::ASTNodeHandle>(
                m_memPoolAcc[0].Alloc<Compiler::LambdaNode>(0, branch));
            auto *bindingsData =
                m_memPoolAcc[0].derefHandle(derefd_bind_node.m_bindings);
            bindingsData[0] = no_arg_lambda;
            auto &derefd_root = *m_memPoolAcc[0].derefHandle(m_root);
            Compiler::ASTNodeHandle result;
            if (derefd_root.m_type == Compiler::ASTNode::Type::Bind) {
              auto &root_as_bind_node =
                  static_cast<Compiler::BindNode &>(derefd_root);
              const auto prev_child = root_as_bind_node.m_childExpr;
              root_as_bind_node.m_childExpr = bind_node_for_lambda;
              derefd_bind_node.m_childExpr = prev_child;
              result = m_root;
            } else {
              derefd_bind_node.m_childExpr = m_root;
              result = bind_node_for_lambda;
            }
            const auto ident_0 =
                m_memPoolAcc[0].Alloc<Compiler::IdentifierNode>(0);
            const auto call_to_new_lambda =
                m_memPoolAcc[0].Alloc<Compiler::CallNode>(0, ident_0,
                                                          m_memPoolAcc);
            branch = call_to_new_lambda;
            return result;
          }
          default:
            return m_root;
          }
        };
        m_root = maybe_rewrite_branch_as_call(ifNode.m_then);
        m_root = maybe_rewrite_branch_as_call(ifNode.m_else);
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

Compiler::ASTNodeHandle BlockPrep::substitute_identifiers_in_range_with_call(
    Compiler::ASTNodeHandle root, const Index_t start, const Index_t end,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  return visit(
      *mem_pool_acc[0].derefHandle(root),
      Visitor{
          [&](Compiler::IdentifierNode &id_node) -> Compiler::ASTNodeHandle {
            if (id_node.m_index >= start && id_node.m_index < end) {
              const auto num_args = end - start;
              auto call_handle = mem_pool_acc[0].Alloc<Compiler::CallNode>(
                  num_args, root, mem_pool_acc);
              auto &derefd_call = *mem_pool_acc[0].derefHandle(call_handle);
              auto *args_data = mem_pool_acc[0].derefHandle(derefd_call.m_args);
              for (Index_t i = 0; i < num_args; ++i) {
                args_data[i] = mem_pool_acc[0].Alloc<Compiler::IdentifierNode>(
                    num_args - i - 1 + start);
              }
              return call_handle;
            }
            return root;
          },
          [&](Compiler::BindNode &bind_node) {
            auto cur_start = start;
            auto cur_end = end;
            if (bind_node.is_rec()) {
              cur_start += bind_node.m_bindings.GetCount();
              cur_end += bind_node.m_bindings.GetCount();
            }
            auto *bindings_data =
                mem_pool_acc[0].derefHandle(bind_node.m_bindings);
            for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
              bindings_data[i] = substitute_identifiers_in_range_with_call(
                  bindings_data[i], cur_start, cur_end, mem_pool_acc);
            }
            if (!bind_node.is_rec()) {
              cur_start += bind_node.m_bindings.GetCount();
              cur_end += bind_node.m_bindings.GetCount();
            }
            bind_node.m_childExpr = substitute_identifiers_in_range_with_call(
                bind_node.m_childExpr, cur_start, cur_end, mem_pool_acc);
            return root;
          },
          [&](Compiler::LambdaNode &lambda_node) {
            lambda_node.m_childExpr = substitute_identifiers_in_range_with_call(
                lambda_node.m_childExpr, start + lambda_node.m_argCount,
                end + lambda_node.m_argCount, mem_pool_acc);
            return root;
          },
          [&](auto &other_node_type) {
            other_node_type.for_each_sub_expr(
                mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                  handle = substitute_identifiers_in_range_with_call(
                      handle, start, end, mem_pool_acc);
                });
            return root;
          }},
      [](const auto &) -> Compiler::ASTNodeHandle {
        throw std::invalid_argument("Unexpected node");
      });
}

Compiler::ASTNodeHandle BlockPrep::rewrite_letrec_as_let(
    Compiler::ASTNodeHandle root,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  // A recursive binding exposes all bound identifiers in the current scope to
  // the currently bound values. The same structure can be replicated by
  // wrapping each bound expression in a lambda accepting as args all bound
  // identifiers in the current scope. The transformation applied looks like:
  // (letrec ((fact (lambda (x) (if (= x 0) 1 (* x (fact (- x 1)))))))
  //   (fact 5))
  // =>
  // (let ((__fact_0 (lambda (fact fib) (lambda (x) (if (= x 0) 1 (* x ((fact
  // fact fib) (- x 1)))))))
  // (__fib_0 (lambda (fact fib) (lambda (x) (if (< x 2) 1 (+ ((fib fact fib) (-
  // x 2)) ((fib fact fib) (- x 1))))))))
  //  (let ((fact (__fact_0 __fact_0 __fib_0))
  //        (fib (__fib_0 __fact_0 __fib_0)))
  //    (fact 5)))
  return visit(
      *mem_pool_acc[0].derefHandle(root),
      Visitor{
          [&](Compiler::BindNode &bind_node) {
            if (!bind_node.is_rec()) {
              bind_node.for_each_sub_expr(
                  mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                    handle = rewrite_letrec_as_let(handle, mem_pool_acc);
                  });
              return root;
            }
            const auto num_bindings = bind_node.m_bindings.GetCount();
            auto new_bind_node = mem_pool_acc[0].Alloc<Compiler::BindNode>(
                num_bindings, false, mem_pool_acc);
            auto &derefd_new_bind_node =
                *mem_pool_acc[0].derefHandle(new_bind_node);
            auto *new_bindings_data =
                mem_pool_acc[0].derefHandle(derefd_new_bind_node.m_bindings);
            auto *bindings_data =
                mem_pool_acc[0].derefHandle(bind_node.m_bindings);
            for (Index_t i = 0; i < num_bindings; ++i) {
              const auto recursive_idents_substituded =
                  substitute_identifiers_in_range_with_call(
                      bindings_data[i], 0, num_bindings, mem_pool_acc);
              const auto new_lambda =
                  mem_pool_acc[0].Alloc<Compiler::LambdaNode>(
                      num_bindings, recursive_idents_substituded);
              new_bindings_data[i] = new_lambda;
            }
            auto bind_node_for_original_idents =
                mem_pool_acc[0].Alloc<Compiler::BindNode>(num_bindings, false,
                                                          mem_pool_acc);
            auto &derefd_bind_node_for_original_idents =
                *mem_pool_acc[0].derefHandle(bind_node_for_original_idents);
            derefd_bind_node_for_original_idents.m_childExpr =
                bind_node.m_childExpr;
            auto *derefd_bound_idents_data = mem_pool_acc[0].derefHandle(
                derefd_bind_node_for_original_idents.m_bindings);
            for (Index_t i = 0; i < num_bindings; ++i) {
              const auto call_target_ident =
                  mem_pool_acc[0].Alloc<Compiler::IdentifierNode>(num_bindings -
                                                                  i - 1);
              auto call_node_handle = mem_pool_acc[0].Alloc<Compiler::CallNode>(
                  num_bindings, call_target_ident, mem_pool_acc);
              auto &call_node_data =
                  *mem_pool_acc[0].derefHandle(call_node_handle);
              auto *call_args =
                  mem_pool_acc[0].derefHandle(call_node_data.m_args);
              for (Index_t j = 0; j < num_bindings; ++j) {
                call_args[j] = mem_pool_acc[0].Alloc<Compiler::IdentifierNode>(
                    num_bindings - j - 1);
              }
              derefd_bound_idents_data[i] = call_node_handle;
            }

            IncreaseBindingRefIndices(bind_node.m_childExpr, num_bindings,
                                      mem_pool_acc, num_bindings, {});
            derefd_new_bind_node.m_childExpr = bind_node_for_original_idents;
            mem_pool_acc[0].DeallocArray(bind_node.m_bindings);
            mem_pool_acc[0].Dealloc(root);
            return rewrite_letrec_as_let(new_bind_node, mem_pool_acc);
          },
          [&](auto &other_node_type) {
            other_node_type.for_each_sub_expr(
                mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                  handle = rewrite_letrec_as_let(handle, mem_pool_acc);
                });
            return root;
          }},
      [](const auto &) -> Compiler::ASTNodeHandle {
        throw std::invalid_argument("Unexpected node");
      });
}

bool BlockPrep::all_idents_in_range_direct_calls(
    const Compiler::ASTNodeHandle node_handle, const Index_t start_idx,
    const Index_t end_index, PortableMemPool::HostAccessor_t &mem_pool_acc) {
  return visit(
      std::as_const(*mem_pool_acc[0].derefHandle(node_handle)),
      Visitor{
          [&](const Compiler::IdentifierNode &id_node) {
            return id_node.m_index < start_idx || id_node.m_index >= end_index;
          },
          [&](const Compiler::BindNode &bind_node) {
            auto cur_start_idx = start_idx;
            auto cur_end_idx = end_index;
            if (bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.GetCount();
              cur_end_idx += bind_node.m_bindings.GetCount();
            }
            auto *bindings_data =
                mem_pool_acc[0].derefHandle(bind_node.m_bindings);
            for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
              if (!all_idents_in_range_direct_calls(bindings_data[i],
                                                    cur_start_idx, cur_end_idx,
                                                    mem_pool_acc)) {
                return false;
              }
            }
            if (!bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.GetCount();
              cur_end_idx += bind_node.m_bindings.GetCount();
            }
            return all_idents_in_range_direct_calls(bind_node.m_childExpr,
                                                    cur_start_idx, cur_end_idx,
                                                    mem_pool_acc);
          },
          [&](const Compiler::CallNode &call_node) {
            const auto &call_target =
                *mem_pool_acc[0].derefHandle(call_node.m_target);
            if (call_target.m_type != Compiler::ASTNode::Type::Identifier) {
              if (!all_idents_in_range_direct_calls(
                      call_node.m_target, start_idx, end_index, mem_pool_acc)) {
                return false;
              }
            }
            const auto *call_arg_data =
                mem_pool_acc[0].derefHandle(call_node.m_args);
            for (Index_t i = 0; i < call_node.m_args.GetCount(); ++i) {
              if (!all_idents_in_range_direct_calls(call_arg_data[i], start_idx,
                                                    end_index, mem_pool_acc)) {
                return false;
              }
            }
            return true;
          },
          [&](const Compiler::LambdaNode &lambda_node) {
            return all_idents_in_range_direct_calls(
                lambda_node.m_childExpr, start_idx + lambda_node.m_argCount,
                end_index + lambda_node.m_argCount, mem_pool_acc);
          },
          [&](const auto &other_node_type) {
            auto all_in_range_direct = true;
            other_node_type.for_each_sub_expr(
                mem_pool_acc, [&](const Compiler::ASTNodeHandle &handle) {
                  all_in_range_direct =
                      all_in_range_direct &&
                      all_idents_in_range_direct_calls(handle, start_idx,
                                                       end_index, mem_pool_acc);
                });
            return all_in_range_direct;
          }},
      [](const auto &) -> bool {
        throw std::invalid_argument("Unexpected node");
      });
}

Compiler::ASTNodeHandle BlockPrep::extend_call_args_with_binding_identifiers(
    Compiler::ASTNodeHandle root, const Index_t start_idx,
    const Index_t end_index, PortableMemPool::HostAccessor_t &mem_pool_acc) {
  return visit(
      *mem_pool_acc[0].derefHandle(root),
      Visitor{
          [&](Compiler::BindNode &bind_node) -> Compiler::ASTNodeHandle {
            auto cur_start_idx = start_idx;
            auto cur_end_idx = end_index;
            if (bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.GetCount();
              cur_end_idx += bind_node.m_bindings.GetCount();
            }
            auto *bindings_data =
                mem_pool_acc[0].derefHandle(bind_node.m_bindings);
            for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
              bindings_data[i] = extend_call_args_with_binding_identifiers(
                  bindings_data[i], cur_start_idx, cur_end_idx, mem_pool_acc);
            }
            if (!bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.GetCount();
              cur_end_idx += bind_node.m_bindings.GetCount();
            }
            bind_node.m_childExpr = extend_call_args_with_binding_identifiers(
                bind_node.m_childExpr, cur_start_idx, cur_end_idx,
                mem_pool_acc);
            return root;
          },
          [&](Compiler::LambdaNode &lambda_node) -> Compiler::ASTNodeHandle {
            lambda_node.m_childExpr = extend_call_args_with_binding_identifiers(
                lambda_node.m_childExpr, start_idx + lambda_node.m_argCount,
                end_index + lambda_node.m_argCount, mem_pool_acc);
            return root;
          },
          [&](Compiler::CallNode &call_node) -> Compiler::ASTNodeHandle {
            auto &target_node =
                *mem_pool_acc[0].derefHandle(call_node.m_target);
            const auto process_call_recursive = [&] {
              call_node.for_each_sub_expr(
                  mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                    handle = extend_call_args_with_binding_identifiers(
                        handle, start_idx, end_index, mem_pool_acc);
                  });
              return root;
            };
            if (target_node.m_type != Compiler::ASTNode::Type::Identifier) {
              return process_call_recursive();
            }
            const auto &identifier =
                static_cast<const Compiler::IdentifierNode &>(target_node);
            if (identifier.m_index < start_idx ||
                identifier.m_index >= end_index) {
              return process_call_recursive();
            }
            const auto num_extra_args = end_index - start_idx;
            const auto num_args = num_extra_args + call_node.m_args.GetCount();
            auto new_call_node = mem_pool_acc[0].Alloc<Compiler::CallNode>(
                num_args, call_node.m_target, mem_pool_acc);
            auto &new_call_node_ref =
                *mem_pool_acc[0].derefHandle(new_call_node);
            auto *new_call_args =
                mem_pool_acc[0].derefHandle(new_call_node_ref.m_args);
            for (Index_t i = 0; i < num_extra_args; ++i) {
              new_call_args[i] =
                  mem_pool_acc[0].Alloc<Compiler::IdentifierNode>(
                      num_extra_args - i - 1 + start_idx);
            }
            auto *original_call_args =
                mem_pool_acc[0].derefHandle(call_node.m_args);
            std::copy(original_call_args,
                      original_call_args + call_node.m_args.GetCount(),
                      new_call_args + num_extra_args);
            mem_pool_acc[0].DeallocArray(call_node.m_args);
            mem_pool_acc[0].Dealloc(root);
            new_call_node_ref.for_each_sub_expr(
                mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                  handle = extend_call_args_with_binding_identifiers(
                      handle, start_idx, end_index, mem_pool_acc);
                });
            return new_call_node;
          },
          [&](auto &other_node_type) {
            other_node_type.for_each_sub_expr(
                mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                  handle = extend_call_args_with_binding_identifiers(
                      handle, start_idx, end_index, mem_pool_acc);
                });
            return root;
          }},
      [](const auto &) -> Compiler::ASTNodeHandle {
        throw std::invalid_argument("Unexpected node");
      });
}

Compiler::ASTNodeHandle BlockPrep::rewrite_recursive_lambdas_with_self_args(
    Compiler::ASTNodeHandle root,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  return visit(
      *mem_pool_acc[0].derefHandle(root),
      Visitor{
          [&](Compiler::BindNode &bind_node) -> Compiler::ASTNodeHandle {
            if (!bind_node.is_rec()) {
              bind_node.for_each_sub_expr(
                  mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                    handle = rewrite_recursive_lambdas_with_self_args(
                        handle, mem_pool_acc);
                  });
              return root;
            }
            const auto is_replacement_candidate = [&] {
              auto *bindings_data =
                  mem_pool_acc[0].derefHandle(bind_node.m_bindings);
              for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
                const auto &binding =
                    *mem_pool_acc[0].derefHandle(bindings_data[i]);
                if (binding.m_type != Compiler::ASTNode::Type::Lambda) {
                  return false;
                }
              }
              if (!all_idents_in_range_direct_calls(
                      bind_node.m_childExpr, 0, bind_node.m_bindings.GetCount(),
                      mem_pool_acc)) {
                return false;
              }
              // Check that all recursive references are direct calls.
              for (Index_t i = 0; i < bind_node.m_bindings.GetCount(); ++i) {
                const auto &binding =
                    *mem_pool_acc[0].derefHandle(bindings_data[i]);
                const auto &lambda_node =
                    static_cast<const Compiler::LambdaNode &>(binding);
                if (!all_idents_in_range_direct_calls(
                        lambda_node.m_childExpr, lambda_node.m_argCount,
                        lambda_node.m_argCount +
                            bind_node.m_bindings.GetCount(),
                        mem_pool_acc)) {
                  return false;
                }
              }
              return true;
            }();
            if (!is_replacement_candidate) {
              bind_node.for_each_sub_expr(
                  mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                    handle = rewrite_recursive_lambdas_with_self_args(
                        handle, mem_pool_acc);
                  });
              return root;
            }
            const auto num_bindings = bind_node.m_bindings.GetCount();
            const auto replacement_bind_node =
                mem_pool_acc[0].Alloc<Compiler::BindNode>(num_bindings, false,
                                                          mem_pool_acc);
            auto &replacment_bind_node_ref =
                *mem_pool_acc[0].derefHandle(replacement_bind_node);
            auto *replacement_bindings = mem_pool_acc[0].derefHandle(
                replacment_bind_node_ref.m_bindings);
            auto *bindings_data =
                mem_pool_acc[0].derefHandle(bind_node.m_bindings);
            for (Index_t i = 0; i < num_bindings; ++i) {
              auto &binding = *mem_pool_acc[0].derefHandle(bindings_data[i]);
              auto &lambda = static_cast<Compiler::LambdaNode &>(binding);
              auto new_lambda = mem_pool_acc[0].Alloc<Compiler::LambdaNode>(
                  lambda.m_argCount + num_bindings, lambda.m_childExpr);
              auto &new_lambda_ref = *mem_pool_acc[0].derefHandle(new_lambda);
              new_lambda_ref.m_childExpr =
                  extend_call_args_with_binding_identifiers(
                      new_lambda_ref.m_childExpr, lambda.m_argCount,
                      lambda.m_argCount + num_bindings, mem_pool_acc);
              mem_pool_acc[0].Dealloc(bindings_data[i]);
              replacement_bindings[i] = new_lambda;
            }
            replacment_bind_node_ref.m_childExpr =
                extend_call_args_with_binding_identifiers(
                    bind_node.m_childExpr, 0, num_bindings, mem_pool_acc);
            mem_pool_acc[0].DeallocArray(bind_node.m_bindings);
            mem_pool_acc[0].Dealloc(root);
            return rewrite_recursive_lambdas_with_self_args(
                replacement_bind_node, mem_pool_acc);
          },
          [&](auto &other_node_type) {
            other_node_type.for_each_sub_expr(
                mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                  handle = rewrite_recursive_lambdas_with_self_args(
                      handle, mem_pool_acc);
                });
            return root;
          }},
      [](const auto &) -> Compiler::ASTNodeHandle {
        throw std::invalid_argument("Unexpected node");
      });
}

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
  const auto with_recursive_lambdas_simplified =
      rewrite_recursive_lambdas_with_self_args(wrapped_in_no_arg_lambda,
                                               memPoolAcc);
  const auto with_letrecs_replaced_by_let =
      rewrite_letrec_as_let(with_recursive_lambdas_simplified, memPoolAcc);
  const auto rewritten_as_prim_ops =
      RewriteAsPrimOps(with_letrecs_replaced_by_let, memPoolAcc);
  return rewritten_as_prim_ops;
}
} // namespace FunGPU
