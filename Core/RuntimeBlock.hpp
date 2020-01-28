#include "Compiler.h"
#include "Error.hpp"
#include "List.hpp"
#include "PortableMemPool.hpp"
#include "Types.h"
#include <CL/sycl.hpp>
#include <atomic>

namespace FunGPU {
template <class DependencyTracker_t>
class RuntimeBlock
    : public PortableMemPool::EnableHandleFromThis<
          RuntimeBlock<DependencyTracker_t>> {
public:
  using SharedRuntimeBlockHandle_t = PortableMemPool::Handle<
      RuntimeBlock<DependencyTracker_t>>;

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
  using DependencyTrackerDeviceAcc_t = cl::sycl::accessor<
    DependencyTracker_t, 1, cl::sycl::access::mode::read_write,
    cl::sycl::access::target::global_buffer>; 

  RuntimeBlock(const Compiler::ASTNodeHandle astNode,
               const SharedRuntimeBlockHandle_t &bindingParent,
               const SharedRuntimeBlockHandle_t &parent,
               const DependencyTrackerDeviceAcc_t &depTracker,
               const RuntimeValueHandle_t &dest,
               const PortableMemPool::DeviceAccessor_t &memPool)
      : m_astNode(astNode), m_bindingParent(bindingParent), m_parent(parent),
        m_depTracker(depTracker), m_dest(dest), m_dependenciesRemainingData(0), m_refCount(0),
        m_memPoolDeviceAcc(memPool),
        m_runtimeValues(memPool) {
    if (m_bindingParent != SharedRuntimeBlockHandle_t()) {
      auto derefdBindingParent = m_memPoolDeviceAcc[0].derefHandle(m_bindingParent);
      derefdBindingParent->IncrementRefCount();
    }
    if (m_parent != SharedRuntimeBlockHandle_t()) {
      auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(m_parent);
      derefdParent->IncrementRefCount();
    }
  }

  struct MapperForFunctionVals {
    MapperForFunctionVals(const PortableMemPool::DeviceAccessor_t &memPoolAcc,
                          const DependencyTrackerDeviceAcc_t& depTracker)
      : m_memPoolAcc(memPoolAcc),
        m_depTracker(depTracker) {}
    void operator()(const RuntimeValue &runtimeValue) {
      if (m_error.GetType() == Error::Type::Success && runtimeValue.m_type == RuntimeValue::Type::Function) {
        auto &funv = runtimeValue.m_data.functionVal;
        if (funv.m_bindingParent != SharedRuntimeBlockHandle_t()) {
          m_error = m_memPoolAcc[0].derefHandle(funv.m_bindingParent)->DecrementRefCount(m_depTracker);
        }
      }
    }
    PortableMemPool::DeviceAccessor_t m_memPoolAcc;
    DependencyTrackerDeviceAcc_t m_depTracker;
    Error m_error;
  };

  Error ClearRefs() {
    if (m_bindingParent != SharedRuntimeBlockHandle_t()) {
      RETURN_IF_FAILURE(m_memPoolDeviceAcc[0].derefHandle(m_bindingParent)->DecrementRefCount(m_depTracker));
    }
    if (m_parent != SharedRuntimeBlockHandle_t()) {
      RETURN_IF_FAILURE(m_memPoolDeviceAcc[0].derefHandle(m_parent)->DecrementRefCount(m_depTracker));
    }
    MapperForFunctionVals mapper(m_memPoolDeviceAcc, m_depTracker);
    m_runtimeValues.map(mapper);
    RETURN_IF_FAILURE(mapper.m_error);
    return Error();
  }

  void IncrementRefCount() {
    cl::sycl::atomic<std::uint32_t> refCountAtomic(
                                                   (cl::sycl::multi_ptr<std::uint32_t, cl::sycl::access::address_space::global_space>(&m_refCount)));
    refCountAtomic.fetch_add(1);
  }

  Error DecrementRefCount(DependencyTrackerDeviceAcc_t depTracker) {
    cl::sycl::atomic<std::uint32_t> refCountAtomic(
        (cl::sycl::multi_ptr<std::uint32_t,
                             cl::sycl::access::address_space::global_space>(
            &m_refCount)));
    const auto prevRefCount = refCountAtomic.fetch_sub(1);
    if (prevRefCount == 1) {
      RETURN_IF_FAILURE(depTracker[0].MarkForDeletion(m_handle));
    }
    return Error();
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

  template<typename... Args>
  Error AllocRuntimeBlockChecked(SharedRuntimeBlockHandle_t& result,  Args&&... args) {
    result = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(std::forward<Args>(args)...);
    if (result == SharedRuntimeBlockHandle_t()) {
      return Error(Error::Type::MemPoolAllocFailure);
    }
    return Error();
  }

  Error PerformEvalPass() {
    auto astNode = GetASTNode();
    switch (astNode->m_type) {
    case Compiler::ASTNode::Type::Bind:
    case Compiler::ASTNode::Type::BindRec: {
      const bool isRec = astNode->m_type == Compiler::ASTNode::Type::BindRec;
      auto bindNode = static_cast<Compiler::BindNode *>(astNode);
      if (m_runtimeValues.size() == 0) {
        auto bindingsData =
            m_memPoolDeviceAcc[0].derefHandle(bindNode->m_bindings);
        for (Index_t i = 0; i < bindNode->m_bindings.GetCount(); ++i) {
          RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
          auto targetRuntimeValue = m_runtimeValues.front();
          SharedRuntimeBlockHandle_t dependencyOnBinding;
          RETURN_IF_FAILURE(AllocRuntimeBlockChecked(dependencyOnBinding, bindingsData[i],
              isRec ? m_handle : m_bindingParent, m_handle, m_depTracker,
                                                     targetRuntimeValue, m_memPoolDeviceAcc));
          RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnBinding));
        }
        IncrementRefCount();
      } else {
        SharedRuntimeBlockHandle_t depOnExpr;
        RETURN_IF_FAILURE(AllocRuntimeBlockChecked(depOnExpr, bindNode->m_childExpr, m_handle, m_parent, m_depTracker,
                                                   m_dest, m_memPoolDeviceAcc));
        RETURN_IF_FAILURE(ClearParent());
        RETURN_IF_FAILURE(AddDependentActiveBlock(depOnExpr, false));
        RETURN_IF_FAILURE(DecrementRefCount(m_depTracker));
      }

      break;
    }
    case Compiler::ASTNode::Type::Call: {
      auto callNode = static_cast<Compiler::CallNode *>(astNode);
      if (m_runtimeValues.size() == 0) {
        auto argsData = m_memPoolDeviceAcc[0].derefHandle(callNode->m_args);
        for (Index_t i = 0; i < callNode->m_args.GetCount(); ++i) {
          RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
          auto targetRuntimeValue = m_runtimeValues.front();
          SharedRuntimeBlockHandle_t dependencyOnArg;
          RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
              dependencyOnArg, argsData[i], m_bindingParent, m_handle,
              m_depTracker, targetRuntimeValue, m_memPoolDeviceAcc));
          RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnArg));
        }
        RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
        SharedRuntimeBlockHandle_t dependencyOnLambda;
        RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
            dependencyOnLambda, callNode->m_target, m_bindingParent, m_handle,
            m_depTracker, m_runtimeValues.front(), m_memPoolDeviceAcc));
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnLambda));
        IncrementRefCount();
      } else {
        RuntimeValue lambdaVal = m_runtimeValues.derefFront();
        m_runtimeValues.pop_front();
        if (lambdaVal.m_type != RuntimeValue::Type::Function) {
          return Error(Error::Type::InvalidType);
        }
        if (lambdaVal.m_data.functionVal.m_argCount !=
            callNode->m_args.GetCount()) {
          return Error(Error::Type::ArityMismatch);
        }
        SharedRuntimeBlockHandle_t lambdaBlock;
        RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
            lambdaBlock, lambdaVal.m_data.functionVal.m_expr, m_handle,
            m_parent, m_depTracker, m_dest, m_memPoolDeviceAcc));
        if (m_bindingParent != SharedRuntimeBlockHandle_t()) {
          RETURN_IF_FAILURE(m_memPoolDeviceAcc[0].derefHandle(m_bindingParent)->DecrementRefCount(m_depTracker));
        }
        m_bindingParent = lambdaVal.m_data.functionVal.m_bindingParent;
        RETURN_IF_FAILURE(AddDependentActiveBlock(lambdaBlock, false));
        RETURN_IF_FAILURE(DecrementRefCount(m_depTracker));
        RETURN_IF_FAILURE(ClearParent());
      }

      break;
    }
    case Compiler::ASTNode::Type::If: {
      auto ifNode = static_cast<Compiler::IfNode *>(astNode);
      if (m_runtimeValues.size() == 0) {
        RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
        SharedRuntimeBlockHandle_t dependencyOnPred;
        RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
            dependencyOnPred, ifNode->m_pred, m_bindingParent, m_handle,
            m_depTracker, m_runtimeValues.front(), m_memPoolDeviceAcc));
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnPred));
        IncrementRefCount();
      } else {
        auto predValue = m_runtimeValues.derefFront();
        m_runtimeValues.pop_front();
        if (predValue.m_type != RuntimeValue::Type::Float_t) {
          return Error(Error::Type::InvalidType);
        }
        const bool isPredTrue = static_cast<bool>(predValue.m_data.floatVal);
        const auto branchToTake = isPredTrue ? ifNode->m_then : ifNode->m_else;
        SharedRuntimeBlockHandle_t dependencyOnBranch;
        RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
            dependencyOnBranch, branchToTake, m_bindingParent, m_parent,
            m_depTracker, m_dest, m_memPoolDeviceAcc));
        RETURN_IF_FAILURE(AddDependentActiveBlock(dependencyOnBranch, false));
        RETURN_IF_FAILURE(DecrementRefCount(m_depTracker));
        RETURN_IF_FAILURE(ClearParent());
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
        RETURN_IF_FAILURE(DecrementRefCount(m_depTracker));
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
      RETURN_IF_FAILURE(m_depTracker[0].MarkForDeletion(m_handle));
      break;
    }
    case Compiler::ASTNode::Type::Identifier: {
      auto identNode = static_cast<Compiler::IdentifierNode *>(astNode);
      Error error;
      const auto identVal = GetRuntimeValueForIndex(identNode->m_index, error);
      RETURN_IF_FAILURE(error);
      RETURN_IF_FAILURE(FillDestValue(identVal->m_type, identVal->m_data));
      RETURN_IF_FAILURE(m_depTracker[0].MarkForDeletion(m_handle));
      break;
    }
    case Compiler::ASTNode::Type::Lambda: {
      auto lambdaNode = static_cast<Compiler::LambdaNode *>(astNode);
      typename RuntimeValue::Data dataVal;
      dataVal.functionVal = FunctionValue(
          lambdaNode->m_childExpr, m_bindingParent, lambdaNode->m_argCount);
      RETURN_IF_FAILURE(FillDestValue(RuntimeValue::Type::Function, dataVal));
      RETURN_IF_FAILURE(m_depTracker[0].MarkForDeletion(m_handle));
      break;
    }
    default:
      return Error(Error::Type::InvalidType);
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
      error = Error(Error::Type::InvalidIndex);
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
      RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
      SharedRuntimeBlockHandle_t dependencyNode;
      RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
          dependencyNode, unaryOp->m_arg0, m_bindingParent, m_handle,
          m_depTracker, m_runtimeValues.front(), m_memPoolDeviceAcc));
      IncrementRefCount();
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
      RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
      SharedRuntimeBlockHandle_t rightNodeBlock;
      RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
          rightNodeBlock, binaryOp->m_arg1, m_bindingParent, m_handle,
          m_depTracker, m_runtimeValues.front(), m_memPoolDeviceAcc));

      RETURN_IF_FAILURE(m_runtimeValues.push_front(RuntimeValue()));
      SharedRuntimeBlockHandle_t leftNodeBlock;
      RETURN_IF_FAILURE(AllocRuntimeBlockChecked(
          leftNodeBlock, binaryOp->m_arg0, m_bindingParent, m_handle,
          m_depTracker, m_runtimeValues.front(), m_memPoolDeviceAcc));
      IncrementRefCount();
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
    m_runtimeValues.pop_front();
    if (argVal.m_type != RuntimeValue::Type::Float_t) {
      return Error(Error::Type::InvalidType);
    }
    typename RuntimeValue::Data dataToSet;
    dataToSet.floatVal = UnaryOpFunctor()(argVal.m_data.floatVal);
    RETURN_IF_FAILURE(DecrementRefCount(m_depTracker));
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
      return Error(Error::Type::InvalidType);
    }
    typename RuntimeValue::Data dataVal;
    dataVal.floatVal =
        BinaryOpFunctor()(lArg.m_data.floatVal, rArg.m_data.floatVal);
    RETURN_IF_FAILURE(DecrementRefCount(m_depTracker));
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
    if (type == RuntimeValue::Type::Function && data.functionVal.m_bindingParent != SharedRuntimeBlockHandle_t()) {
      m_memPoolDeviceAcc[0].derefHandle(data.functionVal.m_bindingParent)->IncrementRefCount();
    }
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

  Error ClearParent() {
    if (m_parent != SharedRuntimeBlockHandle_t()) {
      RETURN_IF_FAILURE(m_memPoolDeviceAcc[0].derefHandle(m_parent)->DecrementRefCount(m_depTracker));
      m_parent = SharedRuntimeBlockHandle_t();
    }
    return Error();
  }

  Compiler::ASTNodeHandle m_astNode;
  List<RuntimeValue> m_runtimeValues;
  SharedRuntimeBlockHandle_t m_bindingParent;
  SharedRuntimeBlockHandle_t m_parent;

  RuntimeValueHandle_t m_dest;
  DependencyTrackerDeviceAcc_t m_depTracker;
  int m_dependenciesRemainingData;
  std::uint32_t m_refCount;
  PortableMemPool::DeviceAccessor_t m_memPoolDeviceAcc;
};
} // namespace FunGPU
