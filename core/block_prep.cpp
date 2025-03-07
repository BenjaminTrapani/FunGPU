#include "core/block_prep.hpp"
#include "core/collect_all_ast_nodes.hpp"
#include "core/visitor.hpp"

namespace FunGPU {
namespace {
template <typename ContainedType>
std::set<std::remove_cvref_t<ContainedType>>
to_concrete(const std::set<ContainedType *> &original) {
  std::set<std::remove_cvref_t<ContainedType>> result;
  for (const auto val : original) {
    result.emplace(*val);
  }
  return result;
}
} // namespace

BlockPrep::BlockPrep(const Index_t registers_per_block,
                     const Index_t instructions_per_cycle,
                     const Index_t cycles_per_block,
                     cl::sycl::buffer<PortableMemPool> pool)
    : m_registers_per_block(registers_per_block),
      m_instructions_per_cycle(instructions_per_cycle),
      m_cycles_per_block(cycles_per_block), m_pool(pool) {}

void BlockPrep::get_prim_ops(Compiler::ASTNodeHandle &root,
                             PortableMemPool::HostAccessor_t mem_pool_acc,
                             std::vector<Compiler::ASTNodeHandle *> &out) {
  struct PrimOpsExtractor {
    PrimOpsExtractor(Compiler::ASTNodeHandle &input_root,
                     PortableMemPool::HostAccessor_t input_mem_pool_acc,
                     std::vector<Compiler::ASTNodeHandle *> &out_vec)
        : m_root(input_root), m_mem_pool_acc(input_mem_pool_acc),
          m_out(out_vec) {}

    bool is_leaf_node(const Compiler::ASTNodeHandle handle) {
      const auto &derefd_node = *m_mem_pool_acc[0].deref_handle(handle);
      const auto type = derefd_node.node_type;
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
      if (node.node_type == Compiler::ASTNode::Type::BindRec) {
        m_out.emplace_back(&m_root);
        return;
      }
      auto *bindings_data = m_mem_pool_acc[0].deref_handle(node.m_bindings);
      bool all_bindings_leaves = false;
      for (size_t i = 0; i < node.m_bindings.get_count() && all_bindings_leaves;
           ++i) {
        all_bindings_leaves = is_leaf_node(bindings_data[i]);
      }
      if (all_bindings_leaves) {
        m_out.emplace_back(&m_root);
        return;
      }
      for (size_t i = 0; i < node.m_bindings.get_count(); ++i) {
        get_prim_ops(bindings_data[i], m_mem_pool_acc, m_out);
      }
    }

    void operator()(Compiler::CallNode &call) {
      auto *call_args_data = m_mem_pool_acc[0].deref_handle(call.m_args);
      if (is_leaf_node(call.m_target)) {
        bool all_args_leaves = true;
        for (std::size_t i = 0; i < call.m_args.get_count() && all_args_leaves;
             ++i) {
          all_args_leaves = is_leaf_node(call_args_data[i]);
        }
        if (all_args_leaves) {
          m_out.emplace_back(&m_root);
          return;
        }
      }
      get_prim_ops(call.m_target, m_mem_pool_acc, m_out);
      for (std::size_t i = 0; i < call.m_args.get_count(); ++i) {
        get_prim_ops(call_args_data[i], m_mem_pool_acc, m_out);
      }
    }

    void operator()(Compiler::BinaryOpNode &binOp) {
      if (is_leaf_node(binOp.m_arg0) && is_leaf_node(binOp.m_arg1)) {
        m_out.emplace_back(&m_root);
      } else {
        get_prim_ops(binOp.m_arg0, m_mem_pool_acc, m_out);
        get_prim_ops(binOp.m_arg1, m_mem_pool_acc, m_out);
      }
    }

    void operator()(Compiler::UnaryOpNode &unaryOp) {
      if (is_leaf_node(unaryOp.m_arg0)) {
        m_out.emplace_back(&m_root);
      } else {
        get_prim_ops(unaryOp.m_arg0, m_mem_pool_acc, m_out);
      }
    }

    void operator()(Compiler::IfNode &ifNode) {
      if (is_leaf_node(ifNode.m_pred)) {
        m_out.emplace_back(&m_root);
      } else {
        get_prim_ops(ifNode.m_pred, m_mem_pool_acc, m_out);
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
    PortableMemPool::HostAccessor_t m_mem_pool_acc;
    Compiler::ASTNodeHandle &m_root;
  } prim_ops_extractor(root, mem_pool_acc, out);

  visit(
      *mem_pool_acc[0].deref_handle(root), prim_ops_extractor,
      [](const auto &node) { throw std::invalid_argument("Unexpected node"); });
}

void BlockPrep::increase_binding_ref_indices(
    const Compiler::ASTNodeHandle root, const std::size_t increment,
    PortableMemPool::HostAccessor_t mem_pool_acc,
    std::size_t min_ref_for_increment,
    const std::set<Compiler::ASTNodeHandle> &idents_to_exclude) {
  visit(
      *mem_pool_acc[0].deref_handle(root),
      Visitor{
          [&](const Compiler::BindNode &node) {
            const auto *bindings_data =
                mem_pool_acc[0].deref_handle(node.m_bindings);
            const auto is_rec =
                node.node_type == Compiler::ASTNode::Type::BindRec;
            for (size_t i = 0; i < node.m_bindings.get_count(); ++i) {
              increase_binding_ref_indices(
                  bindings_data[i], increment, mem_pool_acc,
                  is_rec ? min_ref_for_increment + node.m_bindings.get_count()
                         : min_ref_for_increment,
                  idents_to_exclude);
            }
            increase_binding_ref_indices(
                node.m_child_expr, increment, mem_pool_acc,
                min_ref_for_increment + node.m_bindings.get_count(),
                idents_to_exclude);
          },
          [&](Compiler::IdentifierNode &node) {
            if (node.m_index >= min_ref_for_increment &&
                idents_to_exclude.find(root) == idents_to_exclude.end()) {
              node.m_index += increment;
            }
          },
          [&](const Compiler::LambdaNode &node) {
            increase_binding_ref_indices(
                node.m_child_expr, increment, mem_pool_acc,
                min_ref_for_increment + node.m_arg_count, idents_to_exclude);
          },
          [&](const auto &standard_node) {
            standard_node.for_each_sub_expr(
                mem_pool_acc, [&](const auto handle) {
                  increase_binding_ref_indices(handle, increment, mem_pool_acc,
                                               min_ref_for_increment,
                                               idents_to_exclude);
                });
          }},
      [](const auto &node) { throw std::invalid_argument("Unexpected node"); });
}

Compiler::ASTNodeHandle
BlockPrep::prepare_for_block_generation(Compiler::ASTNodeHandle root) {
  auto mem_pool_acc = m_pool.get_access<cl::sycl::access::mode::read_write>();
  return prepare_for_block_generation(root, mem_pool_acc);
}

Compiler::ASTNodeHandle
BlockPrep::rewrite_as_prim_ops(Compiler::ASTNodeHandle root,
                               PortableMemPool::HostAccessor_t mem_pool_acc) {
  struct RewriteNestedBinaryAsPrimOps {
    RewriteNestedBinaryAsPrimOps(
        const Compiler::ASTNodeHandle input_root,
        PortableMemPool::HostAccessor_t input_mem_pool_acc)
        : m_root(input_root), m_mem_pool_acc(input_mem_pool_acc) {}

    Compiler::ASTNodeHandle
    pull_prim_ops_above(Compiler::ASTNodeHandle &root,
                        std::vector<Compiler::ASTNodeHandle *> &prim_ops) {
      auto bind_node_for_prim_ops =
          m_mem_pool_acc[0].template alloc<Compiler::BindNode>(
              prim_ops.size(), false, m_mem_pool_acc);
      auto &derefd_bind_node_for_prim_ops =
          *m_mem_pool_acc[0].deref_handle(bind_node_for_prim_ops);
      // Only increase references to binding ops if the reference references
      // something above root, otherwise should not increment.
      std::set<Compiler::ASTNodeHandle *> idents_to_exclude;
      for (auto prim_op_handle : prim_ops) {
        collect_all_ast_nodes(*prim_op_handle, m_mem_pool_acc,
                              {Compiler::ASTNode::Type::Identifier},
                              idents_to_exclude);
      }
      increase_binding_ref_indices(root, prim_ops.size(), m_mem_pool_acc, 0,
                                   to_concrete(idents_to_exclude));
      auto *bindings_data = m_mem_pool_acc[0].deref_handle(
          derefd_bind_node_for_prim_ops.m_bindings);
      for (size_t i = 0; i < prim_ops.size(); ++i) {
        bindings_data[i] = *prim_ops[i];
        *prim_ops[i] =
            m_mem_pool_acc[0].template alloc<Compiler::IdentifierNode>(
                prim_ops.size() - i - 1);
      }
      derefd_bind_node_for_prim_ops.m_child_expr =
          rewrite_as_prim_ops(root, m_mem_pool_acc);
      return rewrite_as_prim_ops(bind_node_for_prim_ops, m_mem_pool_acc);
    }

    Compiler::ASTNodeHandle
    pull_prim_ops_under_up(Compiler::ASTNodeHandle &root) {
      std::vector<Compiler::ASTNodeHandle *> prim_ops_under_add;
      get_prim_ops(root, m_mem_pool_acc, prim_ops_under_add);
      // This node is a prim op, no need to rewrite this node
      if (prim_ops_under_add.size() == 1 && prim_ops_under_add[0] == &root) {
        return root;
      }
      return pull_prim_ops_above(root, prim_ops_under_add);
    }

    Compiler::ASTNodeHandle operator()(Compiler::BindNode &node) {
      if (node.node_type != Compiler::ASTNode::Type::Bind) {
        throw std::invalid_argument("Unexpected binding node type");
      }

      std::set<Compiler::ASTNodeHandle *> all_nested_bindings;
      auto *bindings_data = m_mem_pool_acc[0].deref_handle(node.m_bindings);
      std::set<Compiler::ASTNodeHandle *> bound_branches;

      for (size_t i = 0; i < node.m_bindings.get_count(); ++i) {
        std::set<Compiler::ASTNodeHandle *> cur_bindings;
        collect_all_ast_nodes(
            bindings_data[i], m_mem_pool_acc,
            {Compiler::ASTNode::Type::Bind, Compiler::ASTNode::Type::BindRec,
             Compiler::ASTNode::Type::Lambda, Compiler::ASTNode::Type::If},
            cur_bindings);
        // Collect lambdas to prevent collection of binds inside lambda, cannot
        // move these.
        for (auto elem : cur_bindings) {
          const auto type = m_mem_pool_acc[0].deref_handle(*elem)->node_type;
          switch (type) {
          case Compiler::ASTNode::Type::Bind:
          case Compiler::ASTNode::Type::BindRec:
            all_nested_bindings.emplace(elem);
            break;
          case Compiler::ASTNode::Type::Lambda:
            *elem = rewrite_as_prim_ops(*elem, m_mem_pool_acc);
            break;
          case Compiler::ASTNode::Type::If:
            bound_branches.emplace(elem);
            break;
          default:
            throw std::invalid_argument("Unexpected node type");
          }
        }
      }

      if (!all_nested_bindings.empty()) {
        // Move top level of nested let expressions in bound expressions to
        // chain of lets above the current binding expression. Order does not
        // matter, all in the same scope.
        Compiler::ASTNodeHandle most_recent_unested_let;
        const auto outermost_generated_let = **all_nested_bindings.begin();
        for (auto nested_bind_handle : all_nested_bindings) {
          std::set<Compiler::ASTNodeHandle *> idents_to_exclude;
          collect_all_ast_nodes(*nested_bind_handle, m_mem_pool_acc,
                                {Compiler::ASTNode::Type::Identifier},
                                idents_to_exclude);
          auto &nested_bind_expr = static_cast<Compiler::BindNode &>(
              *m_mem_pool_acc[0].deref_handle(*nested_bind_handle));
          increase_binding_ref_indices(
              m_root, nested_bind_expr.m_bindings.get_count(), m_mem_pool_acc,
              0, to_concrete(idents_to_exclude));
          if (most_recent_unested_let != Compiler::ASTNodeHandle()) {
            auto &derefd_most_recent_unested =
                static_cast<Compiler::BindNode &>(
                    *m_mem_pool_acc[0].deref_handle(most_recent_unested_let));
            derefd_most_recent_unested.m_child_expr = *nested_bind_handle;
          }
          const auto tmp_nested_bind_handle = *nested_bind_handle;
          *nested_bind_handle = nested_bind_expr.m_child_expr;
          nested_bind_expr.m_child_expr = m_root;
          most_recent_unested_let = tmp_nested_bind_handle;
        }
        return rewrite_as_prim_ops(outermost_generated_let, m_mem_pool_acc);
      }

      if (!bound_branches.empty()) {
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
        const auto &first_branch_handle = **bound_branches.begin();
        const auto position_of_first_branch_handle =
            &first_branch_handle - bindings_data;
        const auto bind_split_lhs = m_mem_pool_acc[0].alloc<Compiler::BindNode>(
            position_of_first_branch_handle, false, m_mem_pool_acc);
        auto &bind_split_lhs_ref =
            *m_mem_pool_acc[0].deref_handle(bind_split_lhs);
        auto *bind_split_lhs_bindings_data =
            m_mem_pool_acc[0].deref_handle(bind_split_lhs_ref.m_bindings);
        std::copy(bindings_data,
                  bindings_data + position_of_first_branch_handle,
                  bind_split_lhs_bindings_data);
        const auto bind_split_rhs = m_mem_pool_acc[0].alloc<Compiler::BindNode>(
            node.m_bindings.get_count() - position_of_first_branch_handle - 1,
            false, m_mem_pool_acc);
        auto &bind_split_rhs_ref =
            *m_mem_pool_acc[0].deref_handle(bind_split_rhs);
        bind_split_rhs_ref.m_child_expr = node.m_child_expr;
        auto *bind_split_rhs_bindings_data =
            m_mem_pool_acc[0].deref_handle(bind_split_rhs_ref.m_bindings);
        std::copy(bindings_data + position_of_first_branch_handle + 1,
                  bindings_data + node.m_bindings.get_count(),
                  bind_split_rhs_bindings_data);
        for (size_t i = 0; i < bind_split_rhs_ref.m_bindings.get_count(); ++i) {
          increase_binding_ref_indices(
              bind_split_rhs_bindings_data[i],
              bind_split_lhs_ref.m_bindings.get_count() + 1, m_mem_pool_acc, 0,
              {});
        }
        increase_binding_ref_indices(
            first_branch_handle, bind_split_lhs_ref.m_bindings.get_count() + 1,
            m_mem_pool_acc, 0, {});

        const auto continuation_lambda =
            m_mem_pool_acc[0].alloc<Compiler::LambdaNode>(1, bind_split_rhs);
        const auto bind_node_for_continuation =
            m_mem_pool_acc[0].alloc<Compiler::BindNode>(1, false,
                                                        m_mem_pool_acc);
        auto &bind_node_for_continuation_ref =
            *m_mem_pool_acc[0].deref_handle(bind_node_for_continuation);
        auto *bindings_data = m_mem_pool_acc[0].deref_handle(
            bind_node_for_continuation_ref.m_bindings);
        bindings_data[0] = continuation_lambda;
        bind_split_lhs_ref.m_child_expr = bind_node_for_continuation;
        bind_node_for_continuation_ref.m_child_expr = first_branch_handle;

        auto &first_branch_data = static_cast<Compiler::IfNode &>(
            *m_mem_pool_acc[0].deref_handle(first_branch_handle));
        auto call_body_cont_with_true_branch =
            m_mem_pool_acc[0].alloc<Compiler::CallNode>(
                1, m_mem_pool_acc[0].alloc<Compiler::IdentifierNode>(0),
                m_mem_pool_acc);
        auto &call_true_data = static_cast<Compiler::CallNode &>(
            *m_mem_pool_acc[0].deref_handle(call_body_cont_with_true_branch));
        m_mem_pool_acc[0].deref_handle(call_true_data.m_args)[0] =
            first_branch_data.m_then;
        auto call_body_cont_with_false_branch =
            m_mem_pool_acc[0].alloc<Compiler::CallNode>(
                1, m_mem_pool_acc[0].alloc<Compiler::IdentifierNode>(0),
                m_mem_pool_acc);
        auto &call_false_data = static_cast<Compiler::CallNode &>(
            *m_mem_pool_acc[0].deref_handle(call_body_cont_with_false_branch));
        m_mem_pool_acc[0].deref_handle(call_false_data.m_args)[0] =
            first_branch_data.m_else;
        first_branch_data.m_then = call_body_cont_with_true_branch;
        first_branch_data.m_else = call_body_cont_with_false_branch;
        m_mem_pool_acc[0].dealloc_array(node.m_bindings);
        m_mem_pool_acc[0].dealloc(m_root);
        return rewrite_as_prim_ops(bind_split_lhs, m_mem_pool_acc);
      }

      // Decompose complex bound expressions in current scope into their own let
      // expressions.
      std::vector<Compiler::ASTNodeHandle *> prim_ops_for_bindings;
      for (size_t i = 0; i < node.m_bindings.get_count(); ++i) {
        std::vector<Compiler::ASTNodeHandle *> prim_ops_here;
        get_prim_ops(bindings_data[i], m_mem_pool_acc, prim_ops_here);
        if (prim_ops_here.size() > 1 ||
            (prim_ops_here.size() == 1 &&
             prim_ops_here[0] != &bindings_data[i])) {
          prim_ops_for_bindings.insert(prim_ops_for_bindings.end(),
                                       prim_ops_here.begin(),
                                       prim_ops_here.end());
        }
      }
      if (!prim_ops_for_bindings.empty()) {
        return pull_prim_ops_above(m_root, prim_ops_for_bindings);
      }
      // Let expression is completely decomposed, decompose the child expression
      // and return the root.
      node.m_child_expr =
          rewrite_as_prim_ops(node.m_child_expr, m_mem_pool_acc);
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(Compiler::CallNode &call) {
      return pull_prim_ops_under_up(m_root);
    }

    Compiler::ASTNodeHandle operator()(Compiler::IfNode &if_node) {
      std::vector<Compiler::ASTNodeHandle *> prim_ops_under_pred;
      get_prim_ops(m_root, m_mem_pool_acc, prim_ops_under_pred);
      // Branch is already a prim op. Rewrite both sub branches.
      if (prim_ops_under_pred.size() == 1 &&
          prim_ops_under_pred[0] == &m_root) {
        if_node.m_then = rewrite_as_prim_ops(if_node.m_then, m_mem_pool_acc);
        if_node.m_else = rewrite_as_prim_ops(if_node.m_else, m_mem_pool_acc);
        const auto original_root = m_root;
        const auto maybe_rewrite_branch_as_call =
            [&](Compiler::ASTNodeHandle &branch) -> Compiler::ASTNodeHandle {
          // Pred and then can be anything except structures influencing control
          // flow / block layout.
          const auto &branch_node = *m_mem_pool_acc[0].deref_handle(branch);
          switch (branch_node.node_type) {
          case Compiler::ASTNode::Type::Bind:
          case Compiler::ASTNode::Type::BindRec:
          case Compiler::ASTNode::Type::If: {
            // Create no arg lambda containing this block. Pull it above, and
            // generate call to it here.
            auto bind_node_for_lambda =
                m_mem_pool_acc[0].template alloc<Compiler::BindNode>(
                    1, false, m_mem_pool_acc);
            auto &derefd_bind_node =
                *m_mem_pool_acc[0].deref_handle(bind_node_for_lambda);
            // Only increase references to binding ops if the reference
            // references something above root, otherwise should not increment.
            std::set<Compiler::ASTNodeHandle *> idents_to_exclude;
            collect_all_ast_nodes(branch, m_mem_pool_acc,
                                  {Compiler::ASTNode::Type::Identifier},
                                  idents_to_exclude);
            increase_binding_ref_indices(original_root, 1, m_mem_pool_acc, 0,
                                         to_concrete(idents_to_exclude));
            auto no_arg_lambda = static_cast<Compiler::ASTNodeHandle>(
                m_mem_pool_acc[0].alloc<Compiler::LambdaNode>(0, branch));
            auto *bindings_data =
                m_mem_pool_acc[0].deref_handle(derefd_bind_node.m_bindings);
            bindings_data[0] = no_arg_lambda;
            auto &derefd_root = *m_mem_pool_acc[0].deref_handle(m_root);
            Compiler::ASTNodeHandle result;
            if (derefd_root.node_type == Compiler::ASTNode::Type::Bind) {
              auto &root_as_bind_node =
                  static_cast<Compiler::BindNode &>(derefd_root);
              const auto prev_child = root_as_bind_node.m_child_expr;
              root_as_bind_node.m_child_expr = bind_node_for_lambda;
              derefd_bind_node.m_child_expr = prev_child;
              result = m_root;
            } else {
              derefd_bind_node.m_child_expr = m_root;
              result = bind_node_for_lambda;
            }
            const auto ident_0 =
                m_mem_pool_acc[0].alloc<Compiler::IdentifierNode>(0);
            const auto call_to_new_lambda =
                m_mem_pool_acc[0].alloc<Compiler::CallNode>(0, ident_0,
                                                            m_mem_pool_acc);
            branch = call_to_new_lambda;
            return result;
          }
          default:
            return m_root;
          }
        };
        m_root = maybe_rewrite_branch_as_call(if_node.m_then);
        m_root = maybe_rewrite_branch_as_call(if_node.m_else);
        return m_root;
      }

      return pull_prim_ops_under_up(m_root);
    }

    Compiler::ASTNodeHandle operator()(const Compiler::BinaryOpNode &) {
      return pull_prim_ops_under_up(m_root);
    }

    Compiler::ASTNodeHandle operator()(Compiler::UnaryOpNode &node) {
      return pull_prim_ops_under_up(m_root);
    }

    Compiler::ASTNodeHandle operator()(const Compiler::NumberNode &) {
      const auto bind_node =
          m_mem_pool_acc[0].alloc<Compiler::BindNode>(1, false, m_mem_pool_acc);
      auto &derefd_bind_node = *m_mem_pool_acc[0].deref_handle(bind_node);
      auto *binding_data =
          m_mem_pool_acc[0].deref_handle(derefd_bind_node.m_bindings);
      binding_data[0] = m_root;
      derefd_bind_node.m_child_expr =
          m_mem_pool_acc[0].alloc<Compiler::IdentifierNode>(0);
      return bind_node;
    }

    Compiler::ASTNodeHandle operator()(const Compiler::IdentifierNode &) {
      return m_root;
    }

    Compiler::ASTNodeHandle operator()(Compiler::LambdaNode &node) {
      node.m_child_expr =
          rewrite_as_prim_ops(node.m_child_expr, m_mem_pool_acc);
      return m_root;
    }

    Compiler::ASTNodeHandle m_root;
    PortableMemPool::HostAccessor_t m_mem_pool_acc;
  } rewrite_as_prim_ops(root, mem_pool_acc);

  return visit(*mem_pool_acc[0].deref_handle(root), rewrite_as_prim_ops,
               [&](const auto &node) -> Compiler::ASTNodeHandle {
                 throw std::invalid_argument("Unexpected node");
               });
}

Compiler::ASTNodeHandle BlockPrep::substitute_identifiers_in_range_with_call(
    Compiler::ASTNodeHandle root, const Index_t start, const Index_t end,
    PortableMemPool::HostAccessor_t &mem_pool_acc) {
  return visit(
      *mem_pool_acc[0].deref_handle(root),
      Visitor{
          [&](Compiler::IdentifierNode &id_node) -> Compiler::ASTNodeHandle {
            if (id_node.m_index >= start && id_node.m_index < end) {
              const auto num_args = end - start;
              auto call_handle = mem_pool_acc[0].alloc<Compiler::CallNode>(
                  num_args, root, mem_pool_acc);
              auto &derefd_call = *mem_pool_acc[0].deref_handle(call_handle);
              auto *args_data =
                  mem_pool_acc[0].deref_handle(derefd_call.m_args);
              for (Index_t i = 0; i < num_args; ++i) {
                args_data[i] = mem_pool_acc[0].alloc<Compiler::IdentifierNode>(
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
              cur_start += bind_node.m_bindings.get_count();
              cur_end += bind_node.m_bindings.get_count();
            }
            auto *bindings_data =
                mem_pool_acc[0].deref_handle(bind_node.m_bindings);
            for (Index_t i = 0; i < bind_node.m_bindings.get_count(); ++i) {
              bindings_data[i] = substitute_identifiers_in_range_with_call(
                  bindings_data[i], cur_start, cur_end, mem_pool_acc);
            }
            if (!bind_node.is_rec()) {
              cur_start += bind_node.m_bindings.get_count();
              cur_end += bind_node.m_bindings.get_count();
            }
            bind_node.m_child_expr = substitute_identifiers_in_range_with_call(
                bind_node.m_child_expr, cur_start, cur_end, mem_pool_acc);
            return root;
          },
          [&](Compiler::LambdaNode &lambda_node) {
            lambda_node.m_child_expr =
                substitute_identifiers_in_range_with_call(
                    lambda_node.m_child_expr, start + lambda_node.m_arg_count,
                    end + lambda_node.m_arg_count, mem_pool_acc);
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
      *mem_pool_acc[0].deref_handle(root),
      Visitor{
          [&](Compiler::BindNode &bind_node) {
            if (!bind_node.is_rec()) {
              bind_node.for_each_sub_expr(
                  mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                    handle = rewrite_letrec_as_let(handle, mem_pool_acc);
                  });
              return root;
            }
            const auto num_bindings = bind_node.m_bindings.get_count();
            auto new_bind_node = mem_pool_acc[0].alloc<Compiler::BindNode>(
                num_bindings, false, mem_pool_acc);
            auto &derefd_new_bind_node =
                *mem_pool_acc[0].deref_handle(new_bind_node);
            auto *new_bindings_data =
                mem_pool_acc[0].deref_handle(derefd_new_bind_node.m_bindings);
            auto *bindings_data =
                mem_pool_acc[0].deref_handle(bind_node.m_bindings);
            for (Index_t i = 0; i < num_bindings; ++i) {
              const auto recursive_idents_substituded =
                  substitute_identifiers_in_range_with_call(
                      bindings_data[i], 0, num_bindings, mem_pool_acc);
              const auto new_lambda =
                  mem_pool_acc[0].alloc<Compiler::LambdaNode>(
                      num_bindings, recursive_idents_substituded);
              new_bindings_data[i] = new_lambda;
            }
            auto bind_node_for_original_idents =
                mem_pool_acc[0].alloc<Compiler::BindNode>(num_bindings, false,
                                                          mem_pool_acc);
            auto &derefd_bind_node_for_original_idents =
                *mem_pool_acc[0].deref_handle(bind_node_for_original_idents);
            derefd_bind_node_for_original_idents.m_child_expr =
                bind_node.m_child_expr;
            auto *derefd_bound_idents_data = mem_pool_acc[0].deref_handle(
                derefd_bind_node_for_original_idents.m_bindings);
            for (Index_t i = 0; i < num_bindings; ++i) {
              const auto call_target_ident =
                  mem_pool_acc[0].alloc<Compiler::IdentifierNode>(num_bindings -
                                                                  i - 1);
              auto call_node_handle = mem_pool_acc[0].alloc<Compiler::CallNode>(
                  num_bindings, call_target_ident, mem_pool_acc);
              auto &call_node_data =
                  *mem_pool_acc[0].deref_handle(call_node_handle);
              auto *call_args =
                  mem_pool_acc[0].deref_handle(call_node_data.m_args);
              for (Index_t j = 0; j < num_bindings; ++j) {
                call_args[j] = mem_pool_acc[0].alloc<Compiler::IdentifierNode>(
                    num_bindings - j - 1);
              }
              derefd_bound_idents_data[i] = call_node_handle;
            }

            increase_binding_ref_indices(bind_node.m_child_expr, num_bindings,
                                         mem_pool_acc, num_bindings, {});
            derefd_new_bind_node.m_child_expr = bind_node_for_original_idents;
            mem_pool_acc[0].dealloc_array(bind_node.m_bindings);
            mem_pool_acc[0].dealloc(root);
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
      std::as_const(*mem_pool_acc[0].deref_handle(node_handle)),
      Visitor{
          [&](const Compiler::IdentifierNode &id_node) {
            return id_node.m_index < start_idx || id_node.m_index >= end_index;
          },
          [&](const Compiler::BindNode &bind_node) {
            auto cur_start_idx = start_idx;
            auto cur_end_idx = end_index;
            if (bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.get_count();
              cur_end_idx += bind_node.m_bindings.get_count();
            }
            auto *bindings_data =
                mem_pool_acc[0].deref_handle(bind_node.m_bindings);
            for (Index_t i = 0; i < bind_node.m_bindings.get_count(); ++i) {
              if (!all_idents_in_range_direct_calls(bindings_data[i],
                                                    cur_start_idx, cur_end_idx,
                                                    mem_pool_acc)) {
                return false;
              }
            }
            if (!bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.get_count();
              cur_end_idx += bind_node.m_bindings.get_count();
            }
            return all_idents_in_range_direct_calls(bind_node.m_child_expr,
                                                    cur_start_idx, cur_end_idx,
                                                    mem_pool_acc);
          },
          [&](const Compiler::CallNode &call_node) {
            const auto &call_target =
                *mem_pool_acc[0].deref_handle(call_node.m_target);
            if (call_target.node_type != Compiler::ASTNode::Type::Identifier) {
              if (!all_idents_in_range_direct_calls(
                      call_node.m_target, start_idx, end_index, mem_pool_acc)) {
                return false;
              }
            }
            const auto *call_arg_data =
                mem_pool_acc[0].deref_handle(call_node.m_args);
            for (Index_t i = 0; i < call_node.m_args.get_count(); ++i) {
              if (!all_idents_in_range_direct_calls(call_arg_data[i], start_idx,
                                                    end_index, mem_pool_acc)) {
                return false;
              }
            }
            return true;
          },
          [&](const Compiler::LambdaNode &lambda_node) {
            return all_idents_in_range_direct_calls(
                lambda_node.m_child_expr, start_idx + lambda_node.m_arg_count,
                end_index + lambda_node.m_arg_count, mem_pool_acc);
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
      *mem_pool_acc[0].deref_handle(root),
      Visitor{
          [&](Compiler::BindNode &bind_node) -> Compiler::ASTNodeHandle {
            auto cur_start_idx = start_idx;
            auto cur_end_idx = end_index;
            if (bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.get_count();
              cur_end_idx += bind_node.m_bindings.get_count();
            }
            auto *bindings_data =
                mem_pool_acc[0].deref_handle(bind_node.m_bindings);
            for (Index_t i = 0; i < bind_node.m_bindings.get_count(); ++i) {
              bindings_data[i] = extend_call_args_with_binding_identifiers(
                  bindings_data[i], cur_start_idx, cur_end_idx, mem_pool_acc);
            }
            if (!bind_node.is_rec()) {
              cur_start_idx += bind_node.m_bindings.get_count();
              cur_end_idx += bind_node.m_bindings.get_count();
            }
            bind_node.m_child_expr = extend_call_args_with_binding_identifiers(
                bind_node.m_child_expr, cur_start_idx, cur_end_idx,
                mem_pool_acc);
            return root;
          },
          [&](Compiler::LambdaNode &lambda_node) -> Compiler::ASTNodeHandle {
            lambda_node.m_child_expr =
                extend_call_args_with_binding_identifiers(
                    lambda_node.m_child_expr,
                    start_idx + lambda_node.m_arg_count,
                    end_index + lambda_node.m_arg_count, mem_pool_acc);
            return root;
          },
          [&](Compiler::CallNode &call_node) -> Compiler::ASTNodeHandle {
            auto &target_node =
                *mem_pool_acc[0].deref_handle(call_node.m_target);
            const auto process_call_recursive = [&] {
              call_node.for_each_sub_expr(
                  mem_pool_acc, [&](Compiler::ASTNodeHandle &handle) {
                    handle = extend_call_args_with_binding_identifiers(
                        handle, start_idx, end_index, mem_pool_acc);
                  });
              return root;
            };
            if (target_node.node_type != Compiler::ASTNode::Type::Identifier) {
              return process_call_recursive();
            }
            const auto &identifier =
                static_cast<const Compiler::IdentifierNode &>(target_node);
            if (identifier.m_index < start_idx ||
                identifier.m_index >= end_index) {
              return process_call_recursive();
            }
            const auto num_extra_args = end_index - start_idx;
            const auto num_args = num_extra_args + call_node.m_args.get_count();
            auto new_call_node = mem_pool_acc[0].alloc<Compiler::CallNode>(
                num_args, call_node.m_target, mem_pool_acc);
            auto &new_call_node_ref =
                *mem_pool_acc[0].deref_handle(new_call_node);
            auto *new_call_args =
                mem_pool_acc[0].deref_handle(new_call_node_ref.m_args);
            for (Index_t i = 0; i < num_extra_args; ++i) {
              new_call_args[i] =
                  mem_pool_acc[0].alloc<Compiler::IdentifierNode>(
                      num_extra_args - i - 1 + start_idx);
            }
            auto *original_call_args =
                mem_pool_acc[0].deref_handle(call_node.m_args);
            std::copy(original_call_args,
                      original_call_args + call_node.m_args.get_count(),
                      new_call_args + num_extra_args);
            mem_pool_acc[0].dealloc_array(call_node.m_args);
            mem_pool_acc[0].dealloc(root);
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
      *mem_pool_acc[0].deref_handle(root),
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
                  mem_pool_acc[0].deref_handle(bind_node.m_bindings);
              for (Index_t i = 0; i < bind_node.m_bindings.get_count(); ++i) {
                const auto &binding =
                    *mem_pool_acc[0].deref_handle(bindings_data[i]);
                if (binding.node_type != Compiler::ASTNode::Type::Lambda) {
                  return false;
                }
              }
              if (!all_idents_in_range_direct_calls(
                      bind_node.m_child_expr, 0,
                      bind_node.m_bindings.get_count(), mem_pool_acc)) {
                return false;
              }
              // Check that all recursive references are direct calls.
              for (Index_t i = 0; i < bind_node.m_bindings.get_count(); ++i) {
                const auto &binding =
                    *mem_pool_acc[0].deref_handle(bindings_data[i]);
                const auto &lambda_node =
                    static_cast<const Compiler::LambdaNode &>(binding);
                if (!all_idents_in_range_direct_calls(
                        lambda_node.m_child_expr, lambda_node.m_arg_count,
                        lambda_node.m_arg_count +
                            bind_node.m_bindings.get_count(),
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
            const auto num_bindings = bind_node.m_bindings.get_count();
            const auto replacement_bind_node =
                mem_pool_acc[0].alloc<Compiler::BindNode>(num_bindings, false,
                                                          mem_pool_acc);
            auto &replacment_bind_node_ref =
                *mem_pool_acc[0].deref_handle(replacement_bind_node);
            auto *replacement_bindings = mem_pool_acc[0].deref_handle(
                replacment_bind_node_ref.m_bindings);
            auto *bindings_data =
                mem_pool_acc[0].deref_handle(bind_node.m_bindings);
            for (Index_t i = 0; i < num_bindings; ++i) {
              auto &binding = *mem_pool_acc[0].deref_handle(bindings_data[i]);
              auto &lambda = static_cast<Compiler::LambdaNode &>(binding);
              auto new_lambda = mem_pool_acc[0].alloc<Compiler::LambdaNode>(
                  lambda.m_arg_count + num_bindings, lambda.m_child_expr);
              auto &new_lambda_ref = *mem_pool_acc[0].deref_handle(new_lambda);
              new_lambda_ref.m_child_expr =
                  extend_call_args_with_binding_identifiers(
                      new_lambda_ref.m_child_expr, lambda.m_arg_count,
                      lambda.m_arg_count + num_bindings, mem_pool_acc);
              mem_pool_acc[0].dealloc(bindings_data[i]);
              replacement_bindings[i] = new_lambda;
            }
            replacment_bind_node_ref.m_child_expr =
                extend_call_args_with_binding_identifiers(
                    bind_node.m_child_expr, 0, num_bindings, mem_pool_acc);
            mem_pool_acc[0].dealloc_array(bind_node.m_bindings);
            mem_pool_acc[0].dealloc(root);
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
      mem_pool_acc[0].alloc<Compiler::LambdaNode>(0, root);
  const auto call_expr =
      mem_pool_acc[0].alloc<Compiler::CallNode>(0, outer_lambda, mem_pool_acc);
  return call_expr;
}

Compiler::ASTNodeHandle BlockPrep::prepare_for_block_generation(
    Compiler::ASTNodeHandle root,
    PortableMemPool::HostAccessor_t mem_pool_acc) {
  const auto wrapped_in_no_arg_lambda =
      wrap_in_no_arg_lambda(root, mem_pool_acc);
  const auto with_recursive_lambdas_simplified =
      rewrite_recursive_lambdas_with_self_args(wrapped_in_no_arg_lambda,
                                               mem_pool_acc);
  const auto with_letrecs_replaced_by_let =
      rewrite_letrec_as_let(with_recursive_lambdas_simplified, mem_pool_acc);
  const auto rewritten_as_prim_ops =
      rewrite_as_prim_ops(with_letrecs_replaced_by_let, mem_pool_acc);
  return rewritten_as_prim_ops;
}
} // namespace FunGPU
