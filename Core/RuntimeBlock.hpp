#include "Compiler.hpp"
#include "Error.hpp"
#include "GarbageCollector.hpp"
#include "List.hpp"
#include "PortableMemPool.hpp"
#include "Types.hpp"
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
        m_isMarkedData(false), m_numBound(0) {}

  Error Init() {
    auto &astNodeVal = *m_memPoolDeviceAcc[0].derefHandle(m_astNode);
    if (astNodeVal.m_frameSize == 0) {
      m_runtimeValues = PortableMemPool::ArrayHandle<RuntimeValue>();
      return Error();
    }
    m_runtimeValues = m_memPoolDeviceAcc[0].template AllocArray<RuntimeValue>(
                                                        astNodeVal.m_frameSize);
    return m_runtimeValues == PortableMemPool::ArrayHandle<RuntimeValue>() ? Error(Error::Type::MemPoolAllocFailure) : Error();
  }

  ~RuntimeBlock() {
    if (m_runtimeValues != PortableMemPool::ArrayHandle<RuntimeValue>()) {
      m_memPoolDeviceAcc[0].DeallocArray(m_runtimeValues);
    }
  }

  bool SetMarked() {
    cl::sycl::atomic<unsigned int> isMarkedAtomic(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_isMarkedData)));
    return isMarkedAtomic.exchange(true);
  }

  bool ExpandMarkings() {
    bool anyMarkingsExpanded = false;
    if (GetIsMarked()) {
      if (m_bindingParent != SharedRuntimeBlockHandle_t()) {
        anyMarkingsExpanded |=
            !m_memPoolDeviceAcc[0].derefHandle(m_bindingParent)->SetMarked();
      }
      if (m_parent != SharedRuntimeBlockHandle_t()) {
        anyMarkingsExpanded |=
            !m_memPoolDeviceAcc[0].derefHandle(m_parent)->SetMarked();
      }

      if (m_runtimeValues != PortableMemPool::ArrayHandle<RuntimeValue>()) {
        auto *derefdRuntimeValues =
            m_memPoolDeviceAcc[0].derefHandle(m_runtimeValues);
        for (Index_t i = 0; i < m_runtimeValues.GetCount(); ++i) {
          auto &runtimeValue = derefdRuntimeValues[i];
          if (runtimeValue.m_type == RuntimeValue::Type::Function) {
            auto &funv = runtimeValue.m_data.functionVal;
            if (funv.m_bindingParent != SharedRuntimeBlockHandle_t()) {
              anyMarkingsExpanded |= !m_memPoolDeviceAcc[0]
                                          .derefHandle(funv.m_bindingParent)
                ->SetMarked();
            }
          }
        }
      }
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

  Error PerformEvalPass() {
    class DeallocTempStorageOnExit {
    public:
      DeallocTempStorageOnExit(PortableMemPool::DeviceAccessor_t& acc,
                               const PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t> handleToDealloc): m_memPoolAcc(acc), m_handleToDealloc(handleToDealloc) {}
      ~DeallocTempStorageOnExit() {
        m_memPoolAcc[0].DeallocArray(m_handleToDealloc);
      }
    private:
      PortableMemPool::DeviceAccessor_t m_memPoolAcc;
      const PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t> m_handleToDealloc;
    };

    auto astNode = GetASTNode();
    switch (astNode->m_type) {
    case Compiler::ASTNode::Type::Bind:
    case Compiler::ASTNode::Type::BindRec: {
      const bool isRec = astNode->m_type == Compiler::ASTNode::Type::BindRec;
      auto bindNode = static_cast<Compiler::BindNode *>(astNode);
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      if (m_numBound == 0) {
        auto bindingsData =
            m_memPoolDeviceAcc[0].derefHandle(bindNode->m_bindings);
        const auto tempStorageHandle = m_memPoolDeviceAcc[0].AllocArray<SharedRuntimeBlockHandle_t>(bindNode->m_bindings.GetCount());
        if (tempStorageHandle == PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t>()) {
          return Error(Error::Type::MemPoolAllocFailure);
        }
        auto tempStorageData = m_memPoolDeviceAcc[0].derefHandle(tempStorageHandle);
        DeallocTempStorageOnExit deallocOnExit(m_memPoolDeviceAcc, tempStorageHandle);
        for (Index_t i = 0; i < bindNode->m_bindings.GetCount(); ++i) {
          RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
              tempStorageData[i], bindingsData[i],
              isRec ? m_handle : m_bindingParent, m_handle, m_depTracker,
              m_runtimeValues.ElementHandle(i), m_memPoolDeviceAcc,
              m_garbageCollectorHandle));
        }
        for (Index_t i = 0; i < tempStorageHandle.GetCount(); ++i) {
          RETURN_IF_FAILURE(AddDependentActiveBlock(tempStorageData[i]));
        }
        m_numBound = bindNode->m_bindings.GetCount();
      } else {
        SharedRuntimeBlockHandle_t depOnExpr;
        RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
            depOnExpr, bindNode->m_childExpr, m_handle, m_parent, m_depTracker,
            m_dest, m_memPoolDeviceAcc, m_garbageCollectorHandle));
        RETURN_IF_FAILURE(AddDependentActiveBlock(depOnExpr, false));
      }

      break;
    }
    case Compiler::ASTNode::Type::Call: {
      auto callNode = static_cast<Compiler::CallNode *>(astNode);
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      if (m_numBound == 0) {
        auto argsData = m_memPoolDeviceAcc[0].derefHandle(callNode->m_args);
        const auto handleToTempStorage = m_memPoolDeviceAcc[0].AllocArray<SharedRuntimeBlockHandle_t>(callNode->m_args.GetCount());
        if (handleToTempStorage == PortableMemPool::ArrayHandle<SharedRuntimeBlockHandle_t>()) {
          return Error(Error::Type::MemPoolAllocFailure);
        }
        DeallocTempStorageOnExit deallocOnExit(m_memPoolDeviceAcc, handleToTempStorage);
        auto tempStorageData = m_memPoolDeviceAcc[0].derefHandle(handleToTempStorage);
        for (Index_t i = 0; i < callNode->m_args.GetCount(); ++i) {
          RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
              tempStorageData[i], argsData[i], m_bindingParent, m_handle,
              m_depTracker, m_runtimeValues.ElementHandle(i),
              m_memPoolDeviceAcc, m_garbageCollectorHandle));
        }
        SharedRuntimeBlockHandle_t dependencyOnLambda;
        RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
            dependencyOnLambda, callNode->m_target, m_bindingParent, m_handle,
            m_depTracker,
            m_runtimeValues.ElementHandle(callNode->m_args.GetCount()),
            m_memPoolDeviceAcc, m_garbageCollectorHandle));
        for (Index_t i = 0; i < handleToTempStorage.GetCount(); ++i) {
          RETURN_IF_FAILURE(AddDependentActiveBlock(tempStorageData[i]));
        }
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnLambda));
        m_numBound = callNode->m_args.GetCount() + 1;
      } else {
        RuntimeValue lambdaVal = m_memPoolDeviceAcc[0].derefHandle(
            m_runtimeValues)[callNode->m_args.GetCount()];
        if (lambdaVal.m_type != RuntimeValue::Type::Function) {
          return Error(Error::Type::InvalidArgType);
        }
        if (lambdaVal.m_data.functionVal.m_argCount !=
            callNode->m_args.GetCount()) {
          return Error(Error::Type::ArityMismatch);
        }
        SharedRuntimeBlockHandle_t lambdaBlock;
        RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
            lambdaBlock, lambdaVal.m_data.functionVal.m_expr, m_handle,
            m_parent, m_depTracker, m_dest, m_memPoolDeviceAcc,
            m_garbageCollectorHandle));
        m_bindingParent = lambdaVal.m_data.functionVal.m_bindingParent;
        RETURN_IF_FAILURE(AddDependentActiveBlock(lambdaBlock, false));
        // Lambda node should not be found visible for callers to see.
        --m_numBound;
      }

      break;
    }
    case Compiler::ASTNode::Type::If: {
      auto ifNode = static_cast<Compiler::IfNode *>(astNode);
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      if (m_numBound == 0) {
        SharedRuntimeBlockHandle_t dependencyOnPred;
        RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
            dependencyOnPred, ifNode->m_pred, m_bindingParent, m_handle,
            m_depTracker, m_runtimeValues.ElementHandle(0), m_memPoolDeviceAcc,
            m_garbageCollectorHandle));
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnPred));
        ++m_numBound;
      } else {
        const auto predValue =
            m_memPoolDeviceAcc[0].derefHandle(m_runtimeValues)[0];
        if (predValue.m_type != RuntimeValue::Type::Float_t) {
          return Error(Error::Type::InvalidArgType);
        }
        const bool isPredTrue = static_cast<bool>(predValue.m_data.floatVal);
        const auto branchToTake = isPredTrue ? ifNode->m_then : ifNode->m_else;
        SharedRuntimeBlockHandle_t dependencyOnBranch;
        RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
            dependencyOnBranch, branchToTake, m_bindingParent, m_parent,
            m_depTracker, m_dest, m_memPoolDeviceAcc,
            m_garbageCollectorHandle));
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
        const auto runtimeValuesData =
            m_memPoolDeviceAcc[0].derefHandle(m_runtimeValues);
        auto lArg = runtimeValuesData[0];
        auto rArg = runtimeValuesData[1];
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
      return Error(Error::Type::InvalidASTType);
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
      if (derefdParent->m_runtimeValues ==
          PortableMemPool::ArrayHandle<RuntimeValue>()) {
        tempParent = derefdParent->m_bindingParent;
      } else if (index >= derefdParent->m_numBound) {
        index -= derefdParent->m_numBound;
        tempParent = derefdParent->m_bindingParent;
      } else {
        break;
      }
    }
    if (tempParent == SharedRuntimeBlockHandle_t()) {
      error = Error(Error::Type::InvalidIndex);
      return nullptr;
    }

    auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(tempParent);
    // Index is in tempParent's runtime values. It is 'index' from end.
    return &m_memPoolDeviceAcc[0].derefHandle(
        derefdParent->m_runtimeValues)[derefdParent->m_numBound - index - 1];
  }

  Error MaybeAddUnaryOp(bool &added) {
    if (m_numBound == 0) {
      auto unaryOp = static_cast<Compiler::UnaryOpNode *>(GetASTNode());
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      SharedRuntimeBlockHandle_t dependencyNode;
      RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
          dependencyNode, unaryOp->m_arg0, m_bindingParent, m_handle,
          m_depTracker, m_runtimeValues.ElementHandle(0), m_memPoolDeviceAcc,
          m_garbageCollectorHandle));
      RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyNode));
      added = true;
      ++m_numBound;
    } else {
      added = false;
    }

    return Error();
  }

  Error MaybeAddBinaryOp(bool &added) {
    if (m_numBound == 0) {
      auto binaryOp = static_cast<Compiler::BinaryOpNode *>(GetASTNode());
      auto garbageCollector =
          m_memPoolDeviceAcc[0].derefHandle(m_garbageCollectorHandle);
      SharedRuntimeBlockHandle_t rightNodeBlock;
      RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
          rightNodeBlock, binaryOp->m_arg1, m_bindingParent, m_handle,
          m_depTracker, m_runtimeValues.ElementHandle(1), m_memPoolDeviceAcc,
          m_garbageCollectorHandle));

      SharedRuntimeBlockHandle_t leftNodeBlock;
      RETURN_IF_FAILURE(garbageCollector->AllocManaged(m_memPoolDeviceAcc, 
          leftNodeBlock, binaryOp->m_arg0, m_bindingParent, m_handle,
          m_depTracker, m_runtimeValues.ElementHandle(0), m_memPoolDeviceAcc,
          m_garbageCollectorHandle));
      RETURN_IF_FAILURE(AddDependentActiveBlock(rightNodeBlock));
      RETURN_IF_FAILURE(AddDependentActiveBlock(leftNodeBlock));

      added = true;
      ++m_numBound;
    } else {
      added = false;
    }

    return Error();
  }

  template <class UnaryOpFunctor> Error PerformUnaryOp() {
    const auto argVal = m_memPoolDeviceAcc[0].derefHandle(m_runtimeValues)[0];
    if (argVal.m_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidArgType);
    }
    typename RuntimeValue::Data dataToSet;
    dataToSet.floatVal = UnaryOpFunctor()(argVal.m_data.floatVal);
    RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Float_t, dataToSet));

    return Error();
  }

  template <class BinaryOpFunctor> Error PerformBinaryOp() {
    const auto runtimeValuesData =
        m_memPoolDeviceAcc[0].derefHandle(m_runtimeValues);
    const auto lArg = runtimeValuesData[0];
    const auto rArg = runtimeValuesData[1];
    if (lArg.m_type != RuntimeValue::Type::Float_t ||
        rArg.m_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidArgType);
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
  PortableMemPool::ArrayHandle<RuntimeValue> m_runtimeValues;
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
  Index_t m_numBound;
};
} // namespace FunGPU
