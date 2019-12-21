#include "Compiler.h"
#include "GarbageCollector.h"
#include "List.hpp"
#include "PortableMemPool.hpp"
#include "Types.h"
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
        : m_bindingParent(SharedRuntimeBlockHandle_t()), m_argCount(0) {}
    FunctionValue(const Compiler::ASTNodeHandle expr,
                  const SharedRuntimeBlockHandle_t &bindingParent,
                  const Index_t argCount)
        : m_expr(expr), m_bindingParent(bindingParent), m_argCount(argCount) {}
    FunctionValue(const FunctionValue &other)
        : m_expr(other.m_expr), m_bindingParent(other.m_bindingParent),
          m_argCount(other.m_argCount) {}

    bool operator==(const FunctionValue &other) const {
      return m_expr == other.m_expr && m_bindingParent == other.m_bindingParent;
    }

    Compiler::ASTNodeHandle m_expr;
    SharedRuntimeBlockHandle_t m_bindingParent;
    Index_t m_argCount;
  };

  class RuntimeValue {
  public:
    enum class Type {
      Float_t,
      Function,
    };
    union Data {
      Float_t floatVal;
      FunctionValue functionVal;

      Data() : functionVal(FunctionValue()) {}
      Data(const Data &data) : functionVal(data.functionVal) {}
      Data operator=(const Data &other) {
        functionVal = other.functionVal;
        return *this;
      }

      ~Data() {}
    };

    Type m_type;
    Data m_data;

    bool operator==(const RuntimeValue &other) const {
      if (other.m_type != m_type) {
        return false;
      }
      switch (m_type) {
      case Type::Float_t: {
        return m_data.floatVal == other.m_data.floatVal;
      }
      case Type::Function: {
        return m_data.functionVal == other.m_data.functionVal;
      }
      }
      return false;
    }

    void SetValue(const Type type, const Data data) {
      m_type = type;
      m_data = data;
    }
  };

  class Error {
  public:
    enum class Type {
      Success,
      OutOfMemory,
      InvalidType,
      ArityMismatch,
      InvalidIndex,
    };

    Error() : m_type(Type::Success), m_description(nullptr) {}
    Error(const Error &other)
        : m_type(other.m_type), m_description(other.m_description) {}
    Error(const Type type, const char *desc)
        : m_type(type), m_description(desc) {}

    Type GetType() const { return m_type; }

    const char *GetDescription() const { return m_description; }

  private:
    Type m_type;
    const char *m_description;
  };

  using RuntimeValueHandle_t = PortableMemPool::Handle<RuntimeValue>;

  RuntimeBlock(const Compiler::ASTNodeHandle astNode,
               const SharedRuntimeBlockHandle_t &bindingParent,
               const SharedRuntimeBlockHandle_t &parent,
               const cl::sycl::accessor<
                   DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
                   cl::sycl::access::target::global_buffer> &depTracker,
               const RuntimeValueHandle_t &dest,
               const PortableMemPool::DeviceAccessor_t &memPool,
               const PortableMemPool::Handle<GarbageCollector_t> &gc)
      : m_astNode(astNode), m_bindingParent(bindingParent), m_parent(parent),
        m_depTracker(depTracker), m_dest(dest), m_dependenciesRemainingData(0),
        m_memPoolDeviceAcc(memPool), m_garbageCollectorHandle(gc),
        m_runtimeValues(memPool), m_isMarkedData(false) {}

  bool SetMarked() {
    cl::sycl::atomic<unsigned int> isMarkedAtomic(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_isMarkedData)));
    return isMarkedAtomic.exchange(true);
  }

  struct MapperForFunctionVals {
    MapperForFunctionVals(const PortableMemPool::DeviceAccessor_t &memPoolAcc)
        : m_anyMarkingsExpanded(false), m_memPoolAcc(memPoolAcc) {}
    void operator()(const RuntimeValue &runtimeValue) {
      if (runtimeValue.m_type == RuntimeValue::Type::Function) {
        auto &funv = runtimeValue.m_data.functionVal;
        if (funv.m_bindingParent != SharedRuntimeBlockHandle_t()) {
          m_anyMarkingsExpanded =
              m_anyMarkingsExpanded ||
              !m_memPoolAcc[0].derefHandle(funv.m_bindingParent)->SetMarked();
        }
      }
    }
    bool m_anyMarkingsExpanded;
    PortableMemPool::DeviceAccessor_t m_memPoolAcc;
  };

  bool ExpandMarkings() {
    bool anyMarkingsExpanded = false;
    if (GetIsMarked()) {
      if (m_bindingParent != SharedRuntimeBlockHandle_t()) {
        anyMarkingsExpanded =
            anyMarkingsExpanded ||
            !m_memPoolDeviceAcc[0].derefHandle(m_bindingParent)->SetMarked();
      }
      if (m_parent != SharedRuntimeBlockHandle_t()) {
        anyMarkingsExpanded =
            anyMarkingsExpanded ||
            !m_memPoolDeviceAcc[0].derefHandle(m_parent)->SetMarked();
      }

      MapperForFunctionVals mapper(m_memPoolDeviceAcc);
      m_runtimeValues.map(mapper);

      anyMarkingsExpanded = anyMarkingsExpanded || mapper.m_anyMarkingsExpanded;
    }
    return anyMarkingsExpanded;
  }

  bool GetIsMarked() {
    cl::sycl::atomic<unsigned int> isMarkedAtomic(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_isMarkedData)));
    return isMarkedAtomic.load();
  }

  void ClearMarking() {
    cl::sycl::atomic<unsigned int> isMarkedAtomic(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_isMarkedData)));
    isMarkedAtomic.store(false);
  }

  void
  SetResources(const PortableMemPool::DeviceAccessor_t &memPool,
               const cl::sycl::accessor<
                   DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
                   cl::sycl::access::target::global_buffer> &depTracker) {
    m_memPoolDeviceAcc = memPool;
    m_runtimeValues.SetMemPoolAcc(memPool);
    m_depTracker = depTracker;
  }

  Compiler::ASTNode *GetASTNode() {
    return m_memPoolDeviceAcc[0].derefHandle(m_astNode);
  }

#define RETURN_IF_FAILURE(expr)                                                \
  {                                                                            \
    const auto __error = expr;                                                 \
    if (__error.GetType() != Error::Type::Success) {                           \
      return __error;                                                          \
    }                                                                          \
  }

  struct RequiredAllocDesc {
    Index_t runtimeValuesRequired;
    Index_t dependentBlocksRequired;
    RequiredAllocDesc(const Index_t runtimeValuesReq,
                      const Index_t dependentBlocksReq)
        : runtimeValuesRequired(runtimeValuesReq),
          dependentBlocksRequired(dependentBlocksReq) {}
  };

  RequiredAllocDesc GetRequiredAllocs() {
    auto astNode = GetASTNode();
    switch (astNode->m_type) {
    case Compiler::ASTNode::Type::Bind:
    case Compiler::ASTNode::Type::BindRec: {
      const auto bindNode = static_cast<const Compiler::BindNode *>(astNode);
      const auto bindingsCount = bindNode->m_bindings.GetCount();
      return RequiredAllocDesc(bindingsCount, bindingsCount + 1);
    }
    case Compiler::ASTNode::Type::Call: {
      const auto callNode = static_cast<const Compiler::CallNode *>(astNode);
      // One additional arg implicit in lambda target
      const auto argsCount = callNode->m_args.GetCount() + 1;
      // One additional dependency for lambda invocation
      return RequiredAllocDesc(argsCount, argsCount + 1);
    }
    case Compiler::ASTNode::Type::If: {
      return RequiredAllocDesc(1, 2);
    }
    case Compiler::ASTNode::Type::Add:
    case Compiler::ASTNode::Type::Sub:
    case Compiler::ASTNode::Type::Mul:
    case Compiler::ASTNode::Type::Div:
    case Compiler::ASTNode::Type::Equal:
    case Compiler::ASTNode::Type::GreaterThan:
    case Compiler::ASTNode::Type::Remainder:
      return RequiredAllocDesc(2, 2);
    case Compiler::ASTNode::Type::Floor:
      return RequiredAllocDesc(1, 1);
    case Compiler::ASTNode::Type::Number:
    case Compiler::ASTNode::Type::Identifier:
    case Compiler::ASTNode::Type::Lambda:
      return RequiredAllocDesc(0, 0);
    }
    // The eval call will fail, doesn't matter how much space.
    return RequiredAllocDesc(0, 0);
  }

  Error PerformEvalPass() {
    auto astNode = GetASTNode();
    switch (astNode->m_type) {
    case Compiler::ASTNode::Type::Bind:
    case Compiler::ASTNode::Type::BindRec: {
      const bool isRec = astNode->m_type == Compiler::ASTNode::Type::BindRec;
      auto bindNode = static_cast<Compiler::BindNode *>(astNode);
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      if (m_runtimeValues.size() == 0) {
        auto bindingsData =
            m_memPoolDeviceAcc[0].derefHandle(bindNode->m_bindings);
        for (Index_t i = 0; i < bindNode->m_bindings.GetCount(); ++i) {
          if (!m_runtimeValues.push_front(RuntimeValue())) {
            return Error(Error::Type::OutOfMemory,
                         "While allocating space for bindings");
          }
          auto targetRuntimeValue = m_runtimeValues.front();
          auto dependencyOnBinding = garbageCollector->AllocManaged(
              bindingsData[i], isRec ? m_handle : m_bindingParent, m_handle,
              m_depTracker, targetRuntimeValue, m_memPoolDeviceAcc,
              m_garbageCollectorHandle);
          if (dependencyOnBinding == SharedRuntimeBlockHandle_t()) {
            return Error(Error::Type::OutOfMemory,
                         "Failed to alloc dependency on binding");
          }
          RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnBinding));
        }
      } else {
        auto depOnExpr = garbageCollector->AllocManaged(
            bindNode->m_childExpr, m_handle, m_parent, m_depTracker, m_dest,
            m_memPoolDeviceAcc, m_garbageCollectorHandle);
        if (depOnExpr == SharedRuntimeBlockHandle_t()) {
          return Error(Error::Type::OutOfMemory,
                       "Failed to allocate inner expression in bind");
        }
        RETURN_IF_FAILURE(AddDependentActiveBlock(depOnExpr, false));
      }

      break;
    }
    case Compiler::ASTNode::Type::Call: {
      auto callNode = static_cast<Compiler::CallNode *>(astNode);
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      if (m_runtimeValues.size() == 0) {
        auto argsData = m_memPoolDeviceAcc[0].derefHandle(callNode->m_args);
        for (Index_t i = 0; i < callNode->m_args.GetCount(); ++i) {
          if (!m_runtimeValues.push_front(RuntimeValue())) {
            return Error(Error::Type::OutOfMemory,
                         "While allocating call args runtime values");
          }
          auto targetRuntimeValue = m_runtimeValues.front();
          auto dependencyOnArg = garbageCollector->AllocManaged(
              argsData[i], m_bindingParent, m_handle, m_depTracker,
              targetRuntimeValue, m_memPoolDeviceAcc, m_garbageCollectorHandle);
          if (dependencyOnArg == SharedRuntimeBlockHandle_t()) {
            return Error(Error::Type::OutOfMemory, "Failed to alloc call arg");
          }
          RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnArg));
        }

        if (!m_runtimeValues.push_front(RuntimeValue())) {
          return Error(Error::Type::OutOfMemory,
                       "While allocating call lambda runtime value");
        }
        auto dependencyOnLambda = garbageCollector->AllocManaged(
            callNode->m_target, m_bindingParent, m_handle, m_depTracker,
            m_runtimeValues.front(), m_memPoolDeviceAcc,
            m_garbageCollectorHandle);
        if (dependencyOnLambda == SharedRuntimeBlockHandle_t()) {
          return Error(Error::Type::OutOfMemory,
                       "Failed to alloc space for lambda target of call");
        }
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnLambda));
      } else {
        RuntimeValue lambdaVal = m_runtimeValues.derefFront();
        m_runtimeValues.pop_front();
        if (lambdaVal.m_type != RuntimeValue::Type::Function) {
          return Error(Error::Type::InvalidType, "Cannot call non-function");
        }
        if (lambdaVal.m_data.functionVal.m_argCount !=
            callNode->m_args.GetCount()) {
          return Error(Error::Type::ArityMismatch,
                       "Incorrect number of args in call of lambda expr");
        }
        auto lambdaBlock = garbageCollector->AllocManaged(
            lambdaVal.m_data.functionVal.m_expr, m_handle, m_parent,
            m_depTracker, m_dest, m_memPoolDeviceAcc, m_garbageCollectorHandle);
        if (lambdaBlock == SharedRuntimeBlockHandle_t()) {
          return Error(Error::Type::OutOfMemory,
                       "Failed to alloc space for lambda eval of call");
        }
        m_bindingParent = lambdaVal.m_data.functionVal.m_bindingParent;
        RETURN_IF_FAILURE(AddDependentActiveBlock(lambdaBlock, false));
      }

      break;
    }
    case Compiler::ASTNode::Type::If: {
      auto ifNode = static_cast<Compiler::IfNode *>(astNode);
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      if (m_runtimeValues.size() == 0) {
        if (!m_runtimeValues.push_front(RuntimeValue())) {
          return Error(Error::Type::OutOfMemory,
                       "While allocating runtime value for if pred");
        }
        auto dependencyOnPred = garbageCollector->AllocManaged(
            ifNode->m_pred, m_bindingParent, m_handle, m_depTracker,
            m_runtimeValues.front(), m_memPoolDeviceAcc,
            m_garbageCollectorHandle);
        if (dependencyOnPred == SharedRuntimeBlockHandle_t()) {
          return Error(Error::Type::OutOfMemory,
                       "Failed to allocate dependency on pred for if expr");
        }
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnPred));
      } else {
        auto predValue = m_runtimeValues.derefFront();
        m_runtimeValues.pop_front();
        if (predValue.m_type != RuntimeValue::Type::Float_t) {
          return Error(Error::Type::InvalidType,
                       "Float_t is the only supported bolean type");
        }
        const bool isPredTrue = static_cast<bool>(predValue.m_data.floatVal);
        const auto branchToTake = isPredTrue ? ifNode->m_then : ifNode->m_else;
        auto dependencyOnBranch = garbageCollector->AllocManaged(
            branchToTake, m_bindingParent, m_parent, m_depTracker, m_dest,
            m_memPoolDeviceAcc, m_garbageCollectorHandle);
        if (dependencyOnBranch == SharedRuntimeBlockHandle_t()) {
          return Error(Error::Type::OutOfMemory,
                       "Failed to allocate dependency on branch for if expr");
        }
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnBranch, false));
      }
      break;
    }
    case Compiler::ASTNode::Type::Add: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct AddFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l + r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<AddFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Sub: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct SubFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l - r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<SubFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Mul: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct MulFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l * r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<MulFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Div: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct DivFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l / r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<DivFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Equal: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        auto lArg = m_runtimeValues.derefFront();
        m_runtimeValues.pop_front();
        auto rArg = m_runtimeValues.derefFront();
        m_runtimeValues.pop_front();
        const bool areEq = lArg == rArg;
        typename RuntimeValue::Data dataVal;
        dataVal.floatVal = static_cast<Float_t>(areEq);
        RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Float_t, dataVal));
      }
      break;
    }
    case Compiler::ASTNode::Type::GreaterThan: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct GreaterThanFunctor {
          Float_t operator()(const Float_t l, const Float_t r) { return l > r; }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<GreaterThanFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Remainder: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddBinaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct RemainderFunctor {
          Float_t operator()(const Float_t val, const Float_t base) const {
            return cl::sycl::fmod(val, base);
          }
        };
        RETURN_IF_FAILURE(PerformBinaryOp<RemainderFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Floor: {
      bool wasOpAdded;
      RETURN_IF_FAILURE(MaybeAddUnaryOp(wasOpAdded));
      if (!wasOpAdded) {
        struct FloorFunctor {
          Float_t operator()(const Float_t arg) const {
            return cl::sycl::floor(arg);
          }
        };
        RETURN_IF_FAILURE(PerformUnaryOp<FloorFunctor>());
      }
      break;
    }
    case Compiler::ASTNode::Type::Number: {
      auto numNode = static_cast<Compiler::NumberNode *>(astNode);
      typename RuntimeValue::Data data;
      data.floatVal = numNode->m_value;
      RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Float_t, data));
      break;
    }
    case Compiler::ASTNode::Type::Identifier: {
      auto identNode = static_cast<Compiler::IdentifierNode *>(astNode);
      Error error;
      const auto identVal = GetRuntimeValueForIndex(identNode->m_index, error);
      RETURN_IF_FAILURE(error);
      RETURN_IF_FAILURE(FillDestValue(identVal->m_type, identVal->m_data));
      break;
    }
    case Compiler::ASTNode::Type::Lambda: {
      auto lambdaNode = static_cast<Compiler::LambdaNode *>(astNode);
      typename RuntimeValue::Data dataVal;
      dataVal.functionVal = FunctionValue(
          lambdaNode->m_childExpr, m_bindingParent, lambdaNode->m_argCount);
      RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Function, dataVal));
      break;
    }
    default:
      return Error(Error::Type::InvalidType, "Unexpected AST node in eval");
    }

    return Error();
  }

  SharedRuntimeBlockHandle_t m_handle;

private:
  RuntimeValue *GetRuntimeValueForIndex(Index_t index, Error &error) {
    error = Error();

    auto tempParent = m_bindingParent;
    while (tempParent != SharedRuntimeBlockHandle_t()) {
      auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(tempParent);
      if (index >= derefdParent->m_runtimeValues.size()) {
        index -= derefdParent->m_runtimeValues.size();
        tempParent = derefdParent->m_bindingParent;
      } else {
        break;
      }
    }
    if (tempParent == SharedRuntimeBlockHandle_t()) {
      error = Error(Error::Type::InvalidIndex,
                    "Failed to find runtime value for index");
      return nullptr;
    }

    auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(tempParent);
    // Index is in tempParent's runtime values. It is 'index' from beginning.
    const auto resultHandle =
        derefdParent->m_runtimeValues.GetItemAtIndex(index);

    return m_memPoolDeviceAcc[0].derefHandle(resultHandle);
  }

  Error MaybeAddUnaryOp(bool &added) {
    if (m_runtimeValues.size() == 0) {
      auto unaryOp = static_cast<Compiler::UnaryOpNode *>(GetASTNode());
      if (!m_runtimeValues.push_front(RuntimeValue())) {
        return Error(Error::Type::OutOfMemory,
                     "While allocating dest for unary op");
      }
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      auto dependencyNode = garbageCollector->AllocManaged(
          unaryOp->m_arg0, m_bindingParent, m_handle, m_depTracker,
          m_runtimeValues.front(), m_memPoolDeviceAcc,
          m_garbageCollectorHandle);
      if (dependencyNode == SharedRuntimeBlockHandle_t()) {
        return Error(Error::Type::OutOfMemory,
                     "Failed to allocate dependency on arg for unary op");
      }
      RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyNode));
      added = true;
    } else {
      added = false;
    }

    return Error();
  }

  Error MaybeAddBinaryOp(bool &added) {
    if (m_runtimeValues.size() == 0) {
      auto binaryOp = static_cast<Compiler::BinaryOpNode *>(GetASTNode());
      if (!m_runtimeValues.push_front(RuntimeValue())) {
        return Error(Error::Type::OutOfMemory,
                     "While allocating dest for left binary op arg");
      }
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      auto rightNodeBlock = garbageCollector->AllocManaged(
          binaryOp->m_arg1, m_bindingParent, m_handle, m_depTracker,
          m_runtimeValues.front(), m_memPoolDeviceAcc,
          m_garbageCollectorHandle);
      if (rightNodeBlock == SharedRuntimeBlockHandle_t()) {
        return Error(Error::Type::OutOfMemory,
                     "Failed to allocate right arg dep in binary op");
      }

      if (!m_runtimeValues.push_front(RuntimeValue())) {
        return Error(Error::Type::OutOfMemory,
                     "While allocating dest for right binar op arg");
      }
      auto leftNodeBlock = garbageCollector->AllocManaged(
          binaryOp->m_arg0, m_bindingParent, m_handle, m_depTracker,
          m_runtimeValues.front(), m_memPoolDeviceAcc,
          m_garbageCollectorHandle);
      if (leftNodeBlock == SharedRuntimeBlockHandle_t()) {
        return Error(Error::Type::OutOfMemory,
                     "Failed to allocate left arg depn in binary op");
      }

      RETURN_IF_FAILURE(AddDependentActiveBlock(rightNodeBlock));
      RETURN_IF_FAILURE(AddDependentActiveBlock(leftNodeBlock));

      added = true;
    } else {
      added = false;
    }

    return Error();
  }

  template <class UnaryOpFunctor> Error PerformUnaryOp() {
    const auto argVal = m_runtimeValues.derefFront();
    if (argVal.m_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidType, "Expected Float_t in unary op");
    }
    typename RuntimeValue::Data dataToSet;
    dataToSet.floatVal = UnaryOpFunctor()(argVal.m_data.floatVal);
    RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Float_t, dataToSet));

    return Error();
  }

  template <class BinaryOpFunctor> Error PerformBinaryOp() {
    auto lArg = m_runtimeValues.derefFront();
    m_runtimeValues.pop_front();
    auto rArg = m_runtimeValues.derefFront();
    m_runtimeValues.pop_front();
    if (lArg.m_type != RuntimeValue::Type::Float_t ||
        rArg.m_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidType, "Expected Float_ts in binary op");
    }
    typename RuntimeValue::Data dataVal;
    dataVal.floatVal =
        BinaryOpFunctor()(lArg.m_data.floatVal, rArg.m_data.floatVal);
    RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Float_t, dataVal));

    return Error();
  }

  Error AddDependentActiveBlock(const SharedRuntimeBlockHandle_t block,
                                const bool isNewDependency = true) {
    RETURN_IF_FAILURE(m_depTracker[0].AddActiveBlock(block));
    auto derefdBlock = m_memPoolDeviceAcc[0].derefHandle(block);
    if (derefdBlock->m_parent != SharedRuntimeBlockHandle_t() &&
        isNewDependency) {
      auto derefdParent =
          m_memPoolDeviceAcc[0].derefHandle(derefdBlock->m_parent);
      cl::sycl::atomic<int> atomicDepCount(
          (cl::sycl::multi_ptr<int,
                               cl::sycl::access::address_space::global_space>(
              &derefdParent->m_dependenciesRemainingData)));
      atomicDepCount.fetch_add(1);
    }

    return Error();
  }

  Error FillDestValue(const typename RuntimeValue::Type type,
                      const typename RuntimeValue::Data &data) {
    auto destRef = m_memPoolDeviceAcc[0].derefHandle(m_dest);
    destRef->SetValue(type, data);
    if (m_parent != SharedRuntimeBlockHandle_t()) {
      auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(m_parent);
      cl::sycl::atomic<int> atomicDepCount(
          (cl::sycl::multi_ptr<int,
                               cl::sycl::access::address_space::global_space>(
              &derefdParent->m_dependenciesRemainingData)));
      if (atomicDepCount.fetch_add(-1) == 1) {
        RETURN_IF_FAILURE(m_depTracker[0].AddActiveBlock(m_parent));
      }
    }
    return Error();
  }

  Compiler::ASTNodeHandle m_astNode;
  List<RuntimeValue> m_runtimeValues;
  SharedRuntimeBlockHandle_t m_bindingParent;
  SharedRuntimeBlockHandle_t m_parent;

  RuntimeValueHandle_t m_dest;
  cl::sycl::accessor<DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer>
      m_depTracker;
  int m_dependenciesRemainingData;

  PortableMemPool::DeviceAccessor_t m_memPoolDeviceAcc;
  PortableMemPool::Handle<GarbageCollector_t> m_garbageCollectorHandle;

  unsigned int m_isMarkedData;
};
} // namespace FunGPU
