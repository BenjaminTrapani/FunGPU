#include "Compiler.h"
#include "List.hpp"
#include "PortableMemPool.h"
#include "SYCL/sycl.hpp"

#include <atomic>

namespace FunGPU
{
	template <class DependencyTracker_t>
	class RuntimeBlock : public PortableMemPool::EnableHandleFromThis<RuntimeBlock<DependencyTracker_t>>
	{
	public:
		using SharedRuntimeBlockHandle_t = PortableMemPool::Handle<RuntimeBlock<DependencyTracker_t>>;

		class FunctionValue
		{
		public:
			FunctionValue() : m_bindingParent(SharedRuntimeBlockHandle_t()), m_argCount(0) {}
			FunctionValue(const Compiler::ASTNodeHandle expr, const SharedRuntimeBlockHandle_t& bindingParent, const Index_t argCount) :
				m_expr(expr), m_bindingParent(bindingParent), m_argCount(argCount) {}
			FunctionValue(const FunctionValue& other) : m_expr(other.m_expr), m_bindingParent(other.m_bindingParent), 
				m_argCount(other.m_argCount) {}

			bool operator==(const FunctionValue& other) const
			{
				return m_expr == other.m_expr && m_bindingParent == other.m_bindingParent;
			}

			Compiler::ASTNodeHandle m_expr;
			SharedRuntimeBlockHandle_t m_bindingParent;
			Index_t m_argCount;
		};

		class RuntimeValue
		{
		public:
			enum class Type
			{
				Double,
				Function,
			};
			union Data
			{
				double doubleVal;
				FunctionValue functionVal;
				
				Data() : functionVal(FunctionValue()) {}
				Data(const Data& data) : functionVal(data.functionVal) {}
				Data operator=(const Data& other)
				{
					functionVal = other.functionVal;
					return *this;
				}

				~Data() {}
			};

			Type m_type;
			Data m_data;

			bool operator==(const RuntimeValue& other) const
			{
				if (other.m_type != m_type)
				{
					return false;
				}
				switch (m_type)
				{
				case Type::Double:
				{
					return m_data.doubleVal == other.m_data.doubleVal;
				}
				case Type::Function:
				{
					return m_data.functionVal == other.m_data.functionVal;
				}
				default:
				    break;
					//throw std::invalid_argument("Unexpected type in == operator for RuntimeValue");
				}
			}
			void SetValue(const Type type, const Data data)
			{
				m_type = type;
				m_data = data;
			}
		};

		using RuntimeValueHandle_t = PortableMemPool::Handle<RuntimeValue>;

		RuntimeBlock(const Compiler::ASTNodeHandle astNode, const SharedRuntimeBlockHandle_t& bindingParent,
			const SharedRuntimeBlockHandle_t& parent, 
			const cl::sycl::accessor<DependencyTracker_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>& depTracker, 
			const RuntimeValueHandle_t& dest,
			const PortableMemPool::DeviceAccessor_t& memPool) :
			m_astNode(astNode), m_bindingParent(bindingParent), m_parent(parent),
			m_depTracker(depTracker), m_dest(dest), 
			m_dependenciesRemaining(0),
			m_memPoolDeviceAcc(memPool),
			m_runtimeValues(memPool) 
		{
		}

		void SetMemPool(const PortableMemPool::DeviceAccessor_t& memPool)
        {
            m_memPoolDeviceAcc = memPool;
        }

        void SetDependencyTracker(const cl::sycl::accessor<DependencyTracker_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>& depTracker)
        {
		    m_depTracker = depTracker;
        }

		Compiler::ASTNode* GetASTNode()
		{
			return m_memPoolDeviceAcc[0].derefHandle(m_astNode);
		}

		void PerformEvalPass()
		{
			auto astNode = GetASTNode();
			switch (astNode->m_type)
			{
			case Compiler::ASTNode::Type::Bind:
			case Compiler::ASTNode::Type::BindRec:
			{
				const bool isRec = astNode->m_type == Compiler::ASTNode::Type::BindRec;
				auto bindNode = static_cast<Compiler::BindNode*>(astNode);
				if (m_runtimeValues.size() == 0)
				{
				    auto bindingsData = m_memPoolDeviceAcc[0].derefHandle(bindNode->m_bindings);
					for (Index_t i = 0; i < bindNode->m_bindings.GetCount(); ++i)
					{
						m_runtimeValues.push_front(RuntimeValue());
						auto targetRuntimeValue = m_runtimeValues.front();
						auto dependencyOnBinding = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(bindingsData[i],
							isRec ? m_handle : m_bindingParent, m_handle, m_depTracker, targetRuntimeValue, m_memPoolDeviceAcc);
						AddDependentActiveBlock(dependencyOnBinding);
					}
				}
				else
				{
					auto depOnExpr = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(bindNode->m_childExpr, m_handle, m_parent, m_depTracker, m_dest,
						m_memPoolDeviceAcc);
					AddDependentActiveBlock(depOnExpr, false);
				}

				break;
			}
			case Compiler::ASTNode::Type::Call:
			{
				auto callNode = static_cast<Compiler::CallNode*>(astNode);
				if (m_runtimeValues.size() == 0)
				{
				    auto argsData = m_memPoolDeviceAcc[0].derefHandle(callNode->m_args);
					for (Index_t i = 0; i < callNode->m_args.GetCount(); ++i)
					{
						m_runtimeValues.push_front(RuntimeValue());
						auto targetRuntimeValue = m_runtimeValues.front();
						auto dependencyOnArg = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(argsData[i], m_bindingParent, m_handle,
							m_depTracker, targetRuntimeValue, m_memPoolDeviceAcc);
						AddDependentActiveBlock(dependencyOnArg);
					}

					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyOnLambda = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(callNode->m_target, m_bindingParent, m_handle,
						m_depTracker, m_runtimeValues.front(), m_memPoolDeviceAcc);
					AddDependentActiveBlock(dependencyOnLambda);
				}
				else
				{
					RuntimeValue lambdaVal = m_runtimeValues.derefFront();
					m_runtimeValues.pop_front();
					if (lambdaVal.m_type != RuntimeValue::Type::Function)
					{
						//throw std::invalid_argument("Cannot call non-function");
					}
					if (lambdaVal.m_data.functionVal.m_argCount != callNode->m_args.GetCount())
					{
						//throw std::invalid_argument("Incorrect number of args to call of lambda expr");
					}
					auto lambdaBlock = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(lambdaVal.m_data.functionVal.m_expr,
						m_handle, m_parent, m_depTracker, m_dest, m_memPoolDeviceAcc);
					m_bindingParent = lambdaVal.m_data.functionVal.m_bindingParent;
					AddDependentActiveBlock(lambdaBlock, false);
				}

				break;
			}
			case Compiler::ASTNode::Type::If:
			{
				auto ifNode = static_cast<Compiler::IfNode*>(astNode);
				if (m_runtimeValues.size() == 0)
				{
					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyOnPred = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(ifNode->m_pred, m_bindingParent, m_handle, m_depTracker,
						m_runtimeValues.front(), m_memPoolDeviceAcc);
					AddDependentActiveBlock(dependencyOnPred);
				}
				else
				{
					auto predValue = m_runtimeValues.derefFront();
					m_runtimeValues.pop_front();
					if (predValue.m_type != RuntimeValue::Type::Double)
					{
						//throw std::invalid_argument("Double is the only supported boolean type");
					}
					const bool isPredTrue = static_cast<bool>(predValue.m_data.doubleVal);
					const auto branchToTake = isPredTrue ? ifNode->m_then : ifNode->m_else;
					auto dependencyOnBranch = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(branchToTake, m_bindingParent, m_parent, m_depTracker,
						m_dest, m_memPoolDeviceAcc);
					AddDependentActiveBlock(dependencyOnBranch, false);
				}
				break;
			}
			case Compiler::ASTNode::Type::Add:
			{
				if (!MaybeAddBinaryOp())
				{
					struct AddFunctor
					{
						double operator()(const double l, const double r)
						{
							return l + r;
						}
					};
					PerformBinaryOp<AddFunctor>();
				}
				break;
			}
			case Compiler::ASTNode::Type::Sub:
			{
				if (!MaybeAddBinaryOp())
				{
					struct SubFunctor
					{
						double operator()(const double l, const double r)
						{
							return l - r;
						}
					};
					PerformBinaryOp<SubFunctor>();
				}
				break;
			}
			case Compiler::ASTNode::Type::Mul:
			{
				if (!MaybeAddBinaryOp())
				{
					struct MulFunctor
					{
						double operator()(const double l, const double r)
						{
							return l * r;
						}
					};
					PerformBinaryOp<MulFunctor>();
				}
				break;
			}
			case Compiler::ASTNode::Type::Div:
			{
				if (!MaybeAddBinaryOp())
				{
					struct DivFunctor
					{
						double operator()(const double l, const double r)
						{
							return l / r;
						}
					};
					PerformBinaryOp<DivFunctor>();
				}
				break;
			}
			case Compiler::ASTNode::Type::Equal:
			{
				if (!MaybeAddBinaryOp())
				{
					auto lArg = m_runtimeValues.derefFront();
					m_runtimeValues.pop_front();
					auto rArg = m_runtimeValues.derefFront();
					m_runtimeValues.pop_front();
					const bool areEq = lArg == rArg;
					typename RuntimeValue::Data dataVal;
					dataVal.doubleVal = static_cast<double>(areEq);
					FillDestValue(RuntimeValue::Type::Double, dataVal);
				}
				break;
			}
			case Compiler::ASTNode::Type::GreaterThan:
			{
				if (!MaybeAddBinaryOp())
				{
					struct GreaterThanFunctor
					{
						double operator()(const double l, const double r)
						{
							return l > r;
						}
					};
					PerformBinaryOp<GreaterThanFunctor>();
				}
				break;
			}
			case Compiler::ASTNode::Type::Floor:
			{
				auto unaryOp = static_cast<Compiler::UnaryOpNode*>(astNode);
				if (m_runtimeValues.size() == 0)
				{
					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyNode = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(unaryOp->m_arg0, m_bindingParent, m_handle, m_depTracker,
						m_runtimeValues.front(), m_memPoolDeviceAcc);
					AddDependentActiveBlock(dependencyNode);
				}
				else
				{
					const auto argVal = m_runtimeValues.derefFront();
					if (argVal.m_type != RuntimeValue::Type::Double)
					{
						//throw std::invalid_argument("Expected double in floor op");
					}
					typename RuntimeValue::Data dataToSet;
					dataToSet.doubleVal = floor(argVal.m_data.doubleVal);
					FillDestValue(RuntimeValue::Type::Double, dataToSet);
				}
				break;
			}
			case Compiler::ASTNode::Type::Number:
			{
				auto numNode = static_cast<Compiler::NumberNode*>(astNode);
				typename RuntimeValue::Data data;
				data.doubleVal = numNode->m_value;
				FillDestValue(RuntimeValue::Type::Double, data);
				break;
			}
			case Compiler::ASTNode::Type::Identifier:
			{
				auto identNode = static_cast<Compiler::IdentifierNode*>(astNode);
				auto identVal = GetRuntimeValueForIndex(identNode->m_index);
				FillDestValue(identVal->m_type, identVal->m_data);
				break;
			}
			case Compiler::ASTNode::Type::Lambda:
			{
				auto lambdaNode = static_cast<Compiler::LambdaNode*>(astNode);
				typename RuntimeValue::Data dataVal;
				dataVal.functionVal = FunctionValue(lambdaNode->m_childExpr, m_bindingParent, lambdaNode->m_argCount);
				FillDestValue(RuntimeValue::Type::Function, dataVal);
				break;
			}
			default:
			    break;
				//throw std::invalid_argument("Unexpected AST node in eval");
			}
		}

		SharedRuntimeBlockHandle_t m_handle;

	private:
		RuntimeValue* GetRuntimeValueForIndex(Index_t index)
		{
			auto tempParent = m_bindingParent;
			while (tempParent != SharedRuntimeBlockHandle_t())
			{
				auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(tempParent);
				if (index >= derefdParent->m_runtimeValues.size())
				{
					index -= derefdParent->m_runtimeValues.size();
					tempParent = derefdParent->m_bindingParent;
				}
				else
				{
					break;
				}
			}
			if (tempParent == SharedRuntimeBlockHandle_t())
			{
				//throw std::runtime_error("Failed to find runtime value for index");
			}

			auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(tempParent);
			// Index is in tempParent's runtime values. It is 'index' from beginning.
			const auto resultHandle = derefdParent->m_runtimeValues.GetItemAtIndex(index);

			return m_memPoolDeviceAcc[0].derefHandle(resultHandle);
		}

		bool MaybeAddBinaryOp()
		{
			if (m_runtimeValues.size() == 0)
			{
				auto binaryOp = static_cast<Compiler::BinaryOpNode*>(GetASTNode());
				m_runtimeValues.push_front(RuntimeValue());
				auto rightNodeBlock = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(binaryOp->m_arg1, m_bindingParent, m_handle, m_depTracker,
					m_runtimeValues.front(), m_memPoolDeviceAcc);
				m_runtimeValues.push_front(RuntimeValue());
				auto leftNodeBlock = m_memPoolDeviceAcc[0].template Alloc<RuntimeBlock>(binaryOp->m_arg0, m_bindingParent, m_handle, m_depTracker,
					m_runtimeValues.front(), m_memPoolDeviceAcc);
				AddDependentActiveBlock(rightNodeBlock);
				AddDependentActiveBlock(leftNodeBlock);

				return true;
			}
			else
			{
				return false;
			}
		}

		template<class BinaryOpFunctor>
		void PerformBinaryOp()
		{
			auto lArg = m_runtimeValues.derefFront();
			m_runtimeValues.pop_front();
			auto rArg = m_runtimeValues.derefFront();
			m_runtimeValues.pop_front();
			if (lArg.m_type != RuntimeValue::Type::Double || rArg.m_type != RuntimeValue::Type::Double)
			{
				//throw std::invalid_argument("Expected both operands to add to be double");
			}
			typename RuntimeValue::Data dataVal;
			dataVal.doubleVal = BinaryOpFunctor()(lArg.m_data.doubleVal, rArg.m_data.doubleVal);
			FillDestValue(RuntimeValue::Type::Double, dataVal);
		}

		void AddDependentActiveBlock(const SharedRuntimeBlockHandle_t block, const bool isNewDependency = true)
		{
			m_depTracker[0].AddActiveBlock(block);
			auto derefdBlock = m_memPoolDeviceAcc[0].derefHandle(block);
			if (derefdBlock->m_parent != SharedRuntimeBlockHandle_t() && isNewDependency)
			{
				auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(derefdBlock->m_parent);
				++derefdParent->m_dependenciesRemaining;
			}
		}

		void FillDestValue(const typename RuntimeValue::Type type, const typename RuntimeValue::Data& data)
		{
			auto destRef = m_memPoolDeviceAcc[0].derefHandle(m_dest);
			destRef->SetValue(type, data);
			if (m_parent != SharedRuntimeBlockHandle_t())
			{
				auto derefdParent = m_memPoolDeviceAcc[0].derefHandle(m_parent);
				if (--derefdParent->m_dependenciesRemaining == 0)
				{
					m_depTracker[0].AddActiveBlock(m_parent);
				}
			}
		}

		Compiler::ASTNodeHandle m_astNode;
		List<RuntimeValue> m_runtimeValues;
		SharedRuntimeBlockHandle_t m_bindingParent;
		SharedRuntimeBlockHandle_t m_parent;

		RuntimeValueHandle_t m_dest;
        cl::sycl::accessor<DependencyTracker_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> m_depTracker;
		std::atomic<int> m_dependenciesRemaining;

		PortableMemPool::DeviceAccessor_t m_memPoolDeviceAcc;
	};
}
