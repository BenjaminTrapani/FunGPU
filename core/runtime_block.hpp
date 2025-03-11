#include "core/ast_node.hpp"
#include "core/error.hpp"
#include "core/garbage_collector.hpp"
#include "core/list.hpp"
#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"
#include <CL/sycl.hpp>
#include <atomic>

namespace FunGPU {
template <class DependencyTracker_t, Index_t MaxManagedAllocsCount>
class RuntimeBlock
    : public PortableMemPool::EnableHandleFromThis<
          RuntimeBlock<DependencyTracker_t, MaxManagedAllocsCount>> {
public:
  using SharedRuntimeBlockHandle_t = PortableMemPool::Handle<
      RuntimeBlock<DependencyTracker_t, MaxManagedAllocsCount>>;
  using GarbageCollector_t =
      GarbageCollector<RuntimeBlock, MaxManagedAllocsCount>;

  class FunctionValue {
  public:
    FunctionValue()
        : m_binding_parent(SharedRuntimeBlockHandle_t()), m_arg_count(0) {}
    FunctionValue(const ASTNodeHandle expr,
                  const SharedRuntimeBlockHandle_t &binding_parent,
                  const Index_t arg_count)
        : m_expr(expr), m_binding_parent(binding_parent),
          m_arg_count(arg_count) {}
    FunctionValue(const FunctionValue &other)
        : m_expr(other.m_expr), m_binding_parent(other.m_binding_parent),
          m_arg_count(other.m_arg_count) {}

    bool operator==(const FunctionValue &other) const {
      return m_expr == other.m_expr &&
             m_binding_parent == other.m_binding_parent;
    }

    ASTNodeHandle m_expr;
    SharedRuntimeBlockHandle_t m_binding_parent;
    Index_t m_arg_count;
  };

  class RuntimeValue {
  public:
    enum class Type {
      Float_t,
      Function,
    };
    union Data {
      Float_t float_val;
      FunctionValue function_val;

      Data() : function_val(FunctionValue()) {}
      Data(const Data &data) : function_val(data.function_val) {}
      Data operator=(const Data &other) {
        function_val = other.function_val;
        return *this;
      }

      ~Data() {}
    };

    Type node_type;
    Data m_data;

    bool operator==(const RuntimeValue &other) const {
      if (other.node_type != node_type) {
        return false;
      }
      switch (node_type) {
      case Type::Float_t: {
        return m_data.float_val == other.m_data.float_val;
      }
      case Type::Function: {
        return m_data.function_val == other.m_data.function_val;
      }
      }
      return false;
    }

    void SetValue(const Type type, const Data data) {
      node_type = type;
      m_data = data;
    }
  };

  using RuntimeValueHandle_t = PortableMemPool::Handle<RuntimeValue>;

  RuntimeBlock(const ASTNodeHandle ast_node,
               const SharedRuntimeBlockHandle_t &binding_parent,
               const SharedRuntimeBlockHandle_t &parent,
               const cl::sycl::accessor<
                   DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
                   cl::sycl::access::target::global_buffer> &dep_tracker,
               const RuntimeValueHandle_t &dest,
               const PortableMemPool::DeviceAccessor_t &mem_pool,
               const PortableMemPool::Handle<GarbageCollector_t> &gc)
      : m_ast_node(ast_node), m_binding_parent(binding_parent),
        m_parent(parent), m_dep_tracker(dep_tracker), m_dest(dest),
        m_dependencies_remaining_data(0), m_mem_pool_device_acc(mem_pool),
        m_garbage_collector_handle(gc), m_is_marked_data(false),
        m_num_bound(0) {}

  Error init() {
    auto &ast_node_val = *m_mem_pool_device_acc[0].deref_handle(m_ast_node);
    if (ast_node_val.frame_size == 0) {
      m_runtime_values = PortableMemPool::ArrayHandle<RuntimeValue>();
      return Error();
    }
    m_runtime_values =
        m_mem_pool_device_acc[0].template alloc_array<RuntimeValue>(
            ast_node_val.frame_size);
    return m_runtime_values == PortableMemPool::ArrayHandle<RuntimeValue>()
               ? Error(Error::Type::MemPoolAllocFailure)
               : Error();
  }

  ~RuntimeBlock() {
    if (m_runtime_values != PortableMemPool::ArrayHandle<RuntimeValue>()) {
      m_mem_pool_device_acc[0].dealloc_array(m_runtime_values);
    }
  }

  bool set_marked() {
    cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::seq_cst,
                         cl::sycl::memory_scope::device,
                         cl::sycl::access::address_space::global_space>
        is_marked_atomic(m_is_marked_data);
    return is_marked_atomic.exchange(true);
  }

  bool expand_markings() {
    bool any_markings_expanded = false;
    if (get_is_marked()) {
      if (m_binding_parent != SharedRuntimeBlockHandle_t()) {
        any_markings_expanded |= !m_mem_pool_device_acc[0]
                                      .deref_handle(m_binding_parent)
                                      ->set_marked();
      }
      if (m_parent != SharedRuntimeBlockHandle_t()) {
        any_markings_expanded |=
            !m_mem_pool_device_acc[0].deref_handle(m_parent)->set_marked();
      }

      if (m_runtime_values != PortableMemPool::ArrayHandle<RuntimeValue>()) {
        auto *derefd_runtime_values =
            m_mem_pool_device_acc[0].deref_handle(m_runtime_values);
        for (Index_t i = 0; i < m_runtime_values.get_count(); ++i) {
          auto &runtime_value = derefd_runtime_values[i];
          if (runtime_value.node_type == RuntimeValue::Type::Function) {
            auto &funv = runtime_value.m_data.function_val;
            if (funv.m_binding_parent != SharedRuntimeBlockHandle_t()) {
              any_markings_expanded |= !m_mem_pool_device_acc[0]
                                            .deref_handle(funv.m_binding_parent)
                                            ->set_marked();
            }
          }
        }
      }
    }
    return any_markings_expanded;
  }

  bool get_is_marked() {
    cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::seq_cst,
                         cl::sycl::memory_scope::device,
                         cl::sycl::access::address_space::global_space>
        is_marked_atomic(m_is_marked_data);
    return is_marked_atomic.load();
  }

  void clear_marking() { m_is_marked_data = false; }

  void
  set_resources(const PortableMemPool::DeviceAccessor_t &mem_pool,
                const cl::sycl::accessor<
                    DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
                    cl::sycl::access::target::global_buffer> &dep_tracker) {
    m_mem_pool_device_acc = mem_pool;
    m_dep_tracker = dep_tracker;
  }

  ASTNode *GetASTNode() {
    return m_mem_pool_device_acc[0].deref_handle(m_ast_node);
  }

#define RETURN_IF_FAILURE(expr)                                                \
  {                                                                            \
    const auto __error = expr;                                                 \
    if (__error.GetType() != Error::Type::Success) {                           \
      return __error;                                                          \
    }                                                                          \
  }

  Error PerformEvalPass() {
    class DeallocTempStorageOnExit {
    public:
      DeallocTempStorageOnExit(
          PortableMemPool::DeviceAccessor_t &acc,
          const PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t>
              handle_to_dealloc)
          : m_mem_pool_acc(acc), m_handle_to_dealloc(handle_to_dealloc) {}
      ~DeallocTempStorageOnExit() {
        m_mem_pool_acc[0].dealloc_array(m_handle_to_dealloc);
      }

    private:
      PortableMemPool::DeviceAccessor_t m_mem_pool_acc;
      const PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t>
          m_handle_to_dealloc;
    };

    auto ast_node = GetASTNode();
    switch (ast_node->node_type) {
    case ASTNode::Type::Bind:
    case ASTNode::Type::BindRec: {
      const bool is_rec = ast_node->node_type == ASTNode::Type::BindRec;
      auto bind_node = static_cast<BindNode *>(ast_node);
      auto garbage_collector =
          m_mem_pool_device_acc[0].deref_handle(m_garbage_collector_handle);
      if (m_num_bound == 0 && bind_node->m_bindings.get_count() > 0) {
        auto bindings_data =
            m_mem_pool_device_acc[0].deref_handle(bind_node->m_bindings);
        const auto temp_storage_handle =
            m_mem_pool_device_acc[0].alloc_array<SharedRuntimeBlockHandle_t>(
                bind_node->m_bindings.get_count());
        if (temp_storage_handle ==
            PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t>()) {
          return Error(Error::Type::MemPoolAllocFailure);
        }
        auto temp_storage_data =
            m_mem_pool_device_acc[0].deref_handle(temp_storage_handle);
        DeallocTempStorageOnExit dealloc_on_exit(m_mem_pool_device_acc,
                                                 temp_storage_handle);
        for (Index_t i = 0; i < bind_node->m_bindings.get_count(); ++i) {
          RETURN_IF_FAILURE(garbage_collector->alloc_managed(
              m_mem_pool_device_acc, temp_storage_data[i], bindings_data[i],
              is_rec ? m_handle : m_binding_parent, m_handle, m_dep_tracker,
              m_runtime_values.element_handle(i), m_mem_pool_device_acc,
              m_garbage_collector_handle));
        }
        for (Index_t i = 0; i < temp_storage_handle.get_count(); ++i) {
          RETURN_IF_FAILURE(add_dependent_active_block(temp_storage_data[i]));
        }
        m_num_bound = bind_node->m_bindings.get_count();
      } else {
        SharedRuntimeBlockHandle_t dep_on_expr;
        RETURN_IF_FAILURE(garbage_collector->alloc_managed(
            m_mem_pool_device_acc, dep_on_expr, bind_node->m_child_expr,
            m_handle, m_parent, m_dep_tracker, m_dest, m_mem_pool_device_acc,
            m_garbage_collector_handle));
        RETURN_IF_FAILURE(add_dependent_active_block(dep_on_expr, false));
      }

      break;
    }
    case ASTNode::Type::Call: {
      auto call_node = static_cast<CallNode *>(ast_node);
      auto garbage_collector =
          m_mem_pool_device_acc[0].deref_handle(m_garbage_collector_handle);
      if (m_num_bound == 0) {
        if (call_node->m_args.get_count() > 0) {
          auto args_data =
              m_mem_pool_device_acc[0].deref_handle(call_node->m_args);
          const auto handle_to_temp_storage =
              m_mem_pool_device_acc[0].alloc_array<SharedRuntimeBlockHandle_t>(
                  call_node->m_args.get_count());
          if (handle_to_temp_storage ==
              PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t>()) {
            return Error(Error::Type::MemPoolAllocFailure);
          }
          DeallocTempStorageOnExit dealloc_on_exit(m_mem_pool_device_acc,
                                                   handle_to_temp_storage);
          auto temp_storage_data =
              m_mem_pool_device_acc[0].deref_handle(handle_to_temp_storage);
          for (Index_t i = 0; i < call_node->m_args.get_count(); ++i) {
            RETURN_IF_FAILURE(garbage_collector->alloc_managed(
                m_mem_pool_device_acc, temp_storage_data[i], args_data[i],
                m_binding_parent, m_handle, m_dep_tracker,
                m_runtime_values.element_handle(i), m_mem_pool_device_acc,
                m_garbage_collector_handle));
          }
          for (Index_t i = 0; i < handle_to_temp_storage.get_count(); ++i) {
            RETURN_IF_FAILURE(add_dependent_active_block(temp_storage_data[i]));
          }
        }
        SharedRuntimeBlockHandle_t dependency_on_lambda;
        RETURN_IF_FAILURE(garbage_collector->alloc_managed(
            m_mem_pool_device_acc, dependency_on_lambda, call_node->m_target,
            m_binding_parent, m_handle, m_dep_tracker,
            m_runtime_values.element_handle(call_node->m_args.get_count()),
            m_mem_pool_device_acc, m_garbage_collector_handle));
        RETURN_IF_FAILURE(add_dependent_active_block(dependency_on_lambda));
        m_num_bound = call_node->m_args.get_count() + 1;
      } else {
        RuntimeValue lambda_val = m_mem_pool_device_acc[0].deref_handle(
            m_runtime_values)[call_node->m_args.get_count()];
        if (lambda_val.node_type != RuntimeValue::Type::Function) {
          return Error(Error::Type::InvalidArgType);
        }
        if (lambda_val.m_data.function_val.m_arg_count !=
            call_node->m_args.get_count()) {
          return Error(Error::Type::ArityMismatch);
        }
        SharedRuntimeBlockHandle_t lambda_block;
        RETURN_IF_FAILURE(garbage_collector->alloc_managed(
            m_mem_pool_device_acc, lambda_block,
            lambda_val.m_data.function_val.m_expr, m_handle, m_parent,
            m_dep_tracker, m_dest, m_mem_pool_device_acc,
            m_garbage_collector_handle));
        m_binding_parent = lambda_val.m_data.function_val.m_binding_parent;
        RETURN_IF_FAILURE(add_dependent_active_block(lambda_block, false));
        // Lambda node should not be found visible for callers to see.
        --m_num_bound;
      }

      break;
    }
    case ASTNode::Type::If: {
      auto if_node = static_cast<IfNode *>(ast_node);
      auto garbage_collector =
          m_mem_pool_device_acc[0].deref_handle(m_garbage_collector_handle);
      if (m_num_bound == 0) {
        SharedRuntimeBlockHandle_t dependency_on_pred;
        RETURN_IF_FAILURE(garbage_collector->alloc_managed(
            m_mem_pool_device_acc, dependency_on_pred, if_node->m_pred,
            m_binding_parent, m_handle, m_dep_tracker,
            m_runtime_values.element_handle(0), m_mem_pool_device_acc,
            m_garbage_collector_handle));
        RETURN_IF_FAILURE(add_dependent_active_block(dependency_on_pred));
        ++m_num_bound;
      } else {
        const auto pred_value =
            m_mem_pool_device_acc[0].deref_handle(m_runtime_values)[0];
        if (pred_value.node_type != RuntimeValue::Type::Float_t) {
          return Error(Error::Type::InvalidArgType);
        }
        const bool is_pred_true =
            static_cast<bool>(pred_value.m_data.float_val);
        const auto branch_to_take =
            is_pred_true ? if_node->m_then : if_node->m_else;
        SharedRuntimeBlockHandle_t dependency_on_branch;
        RETURN_IF_FAILURE(garbage_collector->alloc_managed(
            m_mem_pool_device_acc, dependency_on_branch, branch_to_take,
            m_binding_parent, m_parent, m_dep_tracker, m_dest,
            m_mem_pool_device_acc, m_garbage_collector_handle));
        RETURN_IF_FAILURE(
            add_dependent_active_block(dependency_on_branch, false));
      }
      break;
    }
    case ASTNode::Type::Add: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct AddFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l + r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<AddFunctor>());
      }
      break;
    }
    case ASTNode::Type::Sub: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct SubFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l - r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<SubFunctor>());
      }
      break;
    }
    case ASTNode::Type::Mul: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct MulFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l * r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<MulFunctor>());
      }
      break;
    }
    case ASTNode::Type::Div: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct DivFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l / r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<DivFunctor>());
      }
      break;
    }
    case ASTNode::Type::Equal: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        const auto runtime_values_data =
            m_mem_pool_device_acc[0].deref_handle(m_runtime_values);
        auto l_arg = runtime_values_data[0];
        auto r_arg = runtime_values_data[1];
        const bool are_eq = l_arg == r_arg;
        typename RuntimeValue::Data data_val;
        data_val.float_val = static_cast<Float_t>(are_eq);
        RETURN_IF_FAILURE(
            fill_dest_value(RuntimeValue::Type::Float_t, data_val));
      }
      break;
    }
    case ASTNode::Type::GreaterThan: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct GreaterThanFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l > r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<GreaterThanFunctor>());
      }
      break;
    }
    case ASTNode::Type::Remainder: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct RemainderFunctor {
          Float_t operator()(const Float_t val, const Float_t base) const {
            return cl::sycl::fmod(val, base);
          }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<RemainderFunctor>());
      }
      break;
    }
    case ASTNode::Type::Expt: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(was_op_added));
      if (!was_op_added) {
        struct ExptFunctor {
          Float_t operator()(const Float_t val, const Float_t power) const {
            return cl::sycl::pow(val, power);
          }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<ExptFunctor>());
      }
      break;
    }
    case ASTNode::Type::Floor: {
      bool was_op_added;
      RETURN_IF_FAILURE(MaybeAddUnaryOp(was_op_added));
      if (!was_op_added) {
        struct FloorFunctor {
          Float_t operator()(const Float_t arg) const {
            return cl::sycl::floor(arg);
          }
        };
        RETURN_IF_FAILURE(PerformUnaryOp<FloorFunctor>());
      }
      break;
    }
    case ASTNode::Type::Number: {
      auto num_node = static_cast<NumberNode *>(ast_node);
      typename RuntimeValue::Data data;
      data.float_val = num_node->m_value;
      RETURN_IF_FAILURE(fill_dest_value(RuntimeValue::Type::Float_t, data));
      break;
    }
    case ASTNode::Type::Identifier: {
      auto ident_node = static_cast<IdentifierNode *>(ast_node);
      Error error;
      const auto ident_val =
          GetRuntimeValueForIndex(ident_node->m_index, error);
      RETURN_IF_FAILURE(error);
      RETURN_IF_FAILURE(
          fill_dest_value(ident_val->node_type, ident_val->m_data));
      break;
    }
    case ASTNode::Type::Lambda: {
      auto lambda_node = static_cast<LambdaNode *>(ast_node);
      typename RuntimeValue::Data data_val;
      data_val.function_val =
          FunctionValue(lambda_node->m_child_expr, m_binding_parent,
                        lambda_node->m_arg_count);
      RETURN_IF_FAILURE(
          fill_dest_value(RuntimeValue::Type::Function, data_val));
      break;
    }
    default:
      return Error(Error::Type::InvalidASTType);
    }

    return Error();
  }

  SharedRuntimeBlockHandle_t m_handle;

private:
  RuntimeValue *GetRuntimeValueForIndex(Index_t index, Error &error) {
    error = Error();

    auto temp_parent = m_binding_parent;
    while (temp_parent != SharedRuntimeBlockHandle_t()) {
      auto derefd_parent = m_mem_pool_device_acc[0].deref_handle(temp_parent);
      if (derefd_parent->m_runtime_values ==
          PortableMemPool::ArrayHandle<RuntimeValue>()) {
        temp_parent = derefd_parent->m_binding_parent;
      } else if (index >= derefd_parent->m_num_bound) {
        index -= derefd_parent->m_num_bound;
        temp_parent = derefd_parent->m_binding_parent;
      } else {
        break;
      }
    }
    if (temp_parent == SharedRuntimeBlockHandle_t()) {
      error = Error(Error::Type::InvalidIndex);
      return nullptr;
    }

    auto derefd_parent = m_mem_pool_device_acc[0].deref_handle(temp_parent);
    // Index is in temp_parent's runtime values. It is 'index' from end.
    return &m_mem_pool_device_acc[0].deref_handle(
        derefd_parent
            ->m_runtime_values)[derefd_parent->m_num_bound - index - 1];
  }

  Error MaybeAddUnaryOp(bool &added) {
    if (m_num_bound == 0) {
      auto unary_op = static_cast<UnaryOpNode *>(GetASTNode());
      auto garbage_collector =
          m_mem_pool_device_acc[0].deref_handle(m_garbage_collector_handle);
      SharedRuntimeBlockHandle_t dependency_node;
      RETURN_IF_FAILURE(garbage_collector->alloc_managed(
          m_mem_pool_device_acc, dependency_node, unary_op->m_arg0,
          m_binding_parent, m_handle, m_dep_tracker,
          m_runtime_values.element_handle(0), m_mem_pool_device_acc,
          m_garbage_collector_handle));
      RETURN_IF_FAILURE(add_dependent_active_block(dependency_node));
      added = true;
      ++m_num_bound;
    } else {
      added = false;
    }

    return Error();
  }

  Error MaybeAddBinaryOp(bool &added) {
    if (m_num_bound == 0) {
      auto binary_op = static_cast<BinaryOpNode *>(GetASTNode());
      auto garbage_collector =
          m_mem_pool_device_acc[0].deref_handle(m_garbage_collector_handle);
      SharedRuntimeBlockHandle_t right_node_block;
      RETURN_IF_FAILURE(garbage_collector->alloc_managed(
          m_mem_pool_device_acc, right_node_block, binary_op->m_arg1,
          m_binding_parent, m_handle, m_dep_tracker,
          m_runtime_values.element_handle(1), m_mem_pool_device_acc,
          m_garbage_collector_handle));

      SharedRuntimeBlockHandle_t left_node_block;
      RETURN_IF_FAILURE(garbage_collector->alloc_managed(
          m_mem_pool_device_acc, left_node_block, binary_op->m_arg0,
          m_binding_parent, m_handle, m_dep_tracker,
          m_runtime_values.element_handle(0), m_mem_pool_device_acc,
          m_garbage_collector_handle));
      RETURN_IF_FAILURE(add_dependent_active_block(right_node_block));
      RETURN_IF_FAILURE(add_dependent_active_block(left_node_block));

      added = true;
      ++m_num_bound;
    } else {
      added = false;
    }

    return Error();
  }

  template <class UnaryOpFunctor> Error PerformUnaryOp() {
    const auto arg_val =
        m_mem_pool_device_acc[0].deref_handle(m_runtime_values)[0];
    if (arg_val.node_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidArgType);
    }
    typename RuntimeValue::Data data_to_set;
    data_to_set.float_val = UnaryOpFunctor()(arg_val.m_data.float_val);
    RETURN_IF_FAILURE(
        fill_dest_value(RuntimeValue::Type::Float_t, data_to_set));

    return Error();
  }

  template <class BinaryOpFunctor> Error PerformBinaryOp() {
    const auto runtime_values_data =
        m_mem_pool_device_acc[0].deref_handle(m_runtime_values);
    const auto l_arg = runtime_values_data[0];
    const auto r_arg = runtime_values_data[1];
    if (l_arg.node_type != RuntimeValue::Type::Float_t ||
        r_arg.node_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidArgType);
    }
    typename RuntimeValue::Data data_val;
    data_val.float_val =
        BinaryOpFunctor()(l_arg.m_data.float_val, r_arg.m_data.float_val);
    RETURN_IF_FAILURE(fill_dest_value(RuntimeValue::Type::Float_t, data_val));

    return Error();
  }

  Error add_dependent_active_block(const SharedRuntimeBlockHandle_t block,
                                   const bool is_new_dependency = true) {
    RETURN_IF_FAILURE(m_dep_tracker[0].add_active_block(block));
    auto derefd_block = m_mem_pool_device_acc[0].deref_handle(block);
    if (derefd_block->m_parent != SharedRuntimeBlockHandle_t() &&
        is_new_dependency) {
      auto derefd_parent =
          m_mem_pool_device_acc[0].deref_handle(derefd_block->m_parent);
      cl::sycl::atomic_ref<int, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          atomic_dep_count(derefd_parent->m_dependencies_remaining_data);
      atomic_dep_count.fetch_add(1);
    }

    return Error();
  }

  Error fill_dest_value(const typename RuntimeValue::Type type,
                        const typename RuntimeValue::Data &data) {
    auto dest_ref = m_mem_pool_device_acc[0].deref_handle(m_dest);
    dest_ref->SetValue(type, data);
    if (m_parent != SharedRuntimeBlockHandle_t()) {
      auto derefd_parent = m_mem_pool_device_acc[0].deref_handle(m_parent);
      cl::sycl::atomic_ref<int, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          atomic_dep_count(derefd_parent->m_dependencies_remaining_data);
      if (atomic_dep_count.fetch_add(-1) == 1) {
        RETURN_IF_FAILURE(m_dep_tracker[0].add_active_block(m_parent));
      }
    }
    return Error();
  }

  ASTNodeHandle m_ast_node;
  PortableMemPool::ArrayHandle<RuntimeValue> m_runtime_values;
  SharedRuntimeBlockHandle_t m_binding_parent;
  SharedRuntimeBlockHandle_t m_parent;

  RuntimeValueHandle_t m_dest;
  cl::sycl::accessor<DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer>
      m_dep_tracker;
  int m_dependencies_remaining_data;

  PortableMemPool::DeviceAccessor_t m_mem_pool_device_acc;
  PortableMemPool::Handle<GarbageCollector_t> m_garbage_collector_handle;

  unsigned int m_is_marked_data;
  Index_t m_num_bound;
};
} // namespace FunGPU
