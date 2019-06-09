#include "Compiler.h"
#include "List.hpp"
#include "PortableMemPool.h"
#include <atomic>

namespace FunGPU
{
	template <class DependencyTracker_t>
	class RuntimeBlock : public PortableMemPool::GetHandleFromThis<RuntimeBlock<DependencyTracker_t>>
	{
	public:
		using RuntimeBlockHandle_t = PortableMemPool::Handle<RuntimeBlock>;

		class FunctionValue
		{
		public:
			FunctionValue() : m_bindingParent(RuntimeBlockHandle_t()), m_argCount(0) {}
			FunctionValue(const Compiler::ASTNodeHandle expr, const RuntimeBlockHandle_t bindingParent, const Index_t argCount) :
				m_expr(expr), m_bindingParent(bindingParent), m_argCount(argCount) {}
			bool operator==(const FunctionValue& other) const
			{
				return m_expr == other.m_expr && m_bindingParent == other.m_bindingParent;
			}

			Compiler::ASTNodeHandle m_expr;
			RuntimeBlockHandle_t m_bindingParent;
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
					throw std::invalid_argument("Unexpected type in == operator for RuntimeValue");
				}
			}
			void SetValue(const Type type, const Data data)
			{
				m_type = type;
				m_data = data;
			}
		};

		RuntimeBlock(const Compiler::ASTNodeHandle astNode, const RuntimeBlockHandle_t bindingParent,
			const RuntimeBlockHandle_t parent, DependencyTracker_t* depTracker, RuntimeValue* dest,
			const std::shared_ptr<PortableMemPool>& memPool) :
			m_astNode(astNode), m_bindingParent(bindingParent), m_parent(parent),
			m_depTracker(depTracker), m_dest(dest), m_memPool(memPool) {
		}

		Compiler::ASTNode* GetASTNode()
		{
			return m_memPool->derefHandle(m_astNode);
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
					for (Index_t i = 0; i < bindNode->m_bindings.size(); ++i)
					{
						m_runtimeValues.push_front(RuntimeValue());
						auto targetRuntimeValue = &m_runtimeValues.front();
						const auto handleToThis = GetHandle();
						auto dependencyOnBinding = m_memPool->Alloc<RuntimeBlock>(bindNode->m_bindings.Get(i),
							isRec ? handleToThis : m_bindingParent, handleToThis, m_depTracker, targetRuntimeValue, m_memPool);
						AddDependentActiveBlock(dependencyOnBinding);
					}
				}
				else
				{
					auto depOnExpr = m_memPool->Alloc<RuntimeBlock>(bindNode->m_childExpr, GetHandle(), m_parent, m_depTracker, m_dest, m_memPool);
					AddDependentActiveBlock(depOnExpr, false);
				}

				break;
			}
			case Compiler::ASTNode::Type::Call:
			{
				auto callNode = static_cast<Compiler::CallNode*>(astNode);
				if (m_runtimeValues.size() == 0)
				{
					for (Index_t i = 0; i < callNode->m_args.size(); ++i)
					{
						m_runtimeValues.push_front(RuntimeValue());
						auto targetRuntimeValue = &m_runtimeValues.front();
						auto dependencyOnArg = m_memPool->Alloc<RuntimeBlock>(callNode->m_args.Get(i), m_bindingParent, GetHandle(),
							m_depTracker, targetRuntimeValue, m_memPool);
						AddDependentActiveBlock(dependencyOnArg);
					}

					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyOnLambda = m_memPool->Alloc<RuntimeBlock>(callNode->m_target, m_bindingParent, GetHandle(),
						m_depTracker, &m_runtimeValues.front(), m_memPool);
					AddDependentActiveBlock(dependencyOnLambda);
				}
				else
				{
					RuntimeValue lambdaVal = m_runtimeValues.front();
					m_runtimeValues.pop_front();
					if (lambdaVal.m_type != RuntimeValue::Type::Function)
					{
						throw std::invalid_argument("Cannot call non-function");
					}
					if (lambdaVal.m_data.functionVal.m_argCount != callNode->m_args.size())
					{
						throw std::invalid_argument("Incorrect number of args to call of lambda expr");
					}
					auto lambdaBlock = m_memPool->Alloc<RuntimeBlock>(lambdaVal.m_data.functionVal.m_expr,
						GetHandle(), m_parent, m_depTracker, m_dest, m_memPool);
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
					auto dependencyOnPred = m_memPool->Alloc<RuntimeBlock>(ifNode->m_pred, m_bindingParent, GetHandle(), m_depTracker,
						&m_runtimeValues.front(), m_memPool);
					AddDependentActiveBlock(dependencyOnPred);
				}
				else
				{
					auto predValue = m_runtimeValues.front();
					m_runtimeValues.pop_front();
					if (predValue.m_type != RuntimeValue::Type::Double)
					{
						throw std::invalid_argument("Double is the only supported boolean type");
					}
					const bool isPredTrue = static_cast<bool>(predValue.m_data.doubleVal);
					const auto branchToTake = isPredTrue ? ifNode->m_then : ifNode->m_else;
					auto dependencyOnBranch = m_memPool->Alloc<RuntimeBlock>(branchToTake, m_bindingParent, m_parent, m_depTracker,
						m_dest, m_memPool);
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
					auto lArg = m_runtimeValues.front();
					m_runtimeValues.pop_front();
					auto rArg = m_runtimeValues.front();
					m_runtimeValues.pop_front();
					const bool areEq = lArg == rArg;
					RuntimeValue::Data dataVal;
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
					auto dependencyNode = m_memPool->Alloc<RuntimeBlock>(unaryOp->m_arg0, m_bindingParent, GetHandle(), m_depTracker,
						&m_runtimeValues.front(), m_memPool);
					AddDependentActiveBlock(dependencyNode);
				}
				else
				{
					const auto argVal = m_runtimeValues.front();
					if (argVal.m_type != RuntimeValue::Type::Double)
					{
						throw std::invalid_argument("Expected double in floor op");
					}
					RuntimeValue::Data dataToSet;
					dataToSet.doubleVal = floor(argVal.m_data.doubleVal);
					FillDestValue(RuntimeValue::Type::Double, dataToSet);
				}
				break;
			}
			case Compiler::ASTNode::Type::Number:
			{
				auto numNode = static_cast<Compiler::NumberNode*>(astNode);
				RuntimeValue::Data data;
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
				RuntimeValue::Data dataVal;
				dataVal.functionVal = FunctionValue(lambdaNode->m_childExpr, m_bindingParent, lambdaNode->m_argCount);
				FillDestValue(RuntimeValue::Type::Function, dataVal);
				break;
			}
			default:
				throw std::invalid_argument("Unexpected AST node in eval");
			}
		}

	private:
		RuntimeValue* GetRuntimeValueForIndex(Index_t index)
		{
			auto tempParent = m_bindingParent;
			while (tempParent != RuntimeBlockHandle_t())
			{
				auto derefdParent = m_memPool->derefHandle(tempParent);
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
			if (tempParent == RuntimeBlockHandle_t())
			{
				throw std::runtime_error("Failed to find runtime value for index");
			}

			auto derefdParent = m_memPool->derefHandle(tempParent);
			// Index is in tempParent's runtime values. It is 'index' from beginning.
			return &derefdParent->m_runtimeValues.GetItemAtIndex(index);
		}

		bool MaybeAddBinaryOp()
		{
			if (m_runtimeValues.size() == 0)
			{
				auto binaryOp = static_cast<Compiler::BinaryOpNode*>(GetASTNode());
				m_runtimeValues.push_front(RuntimeValue());
				auto rightNodeBlock = m_memPool->Alloc<RuntimeBlock>(binaryOp->m_arg1, m_bindingParent, GetHandle(), m_depTracker,
					&m_runtimeValues.front(), m_memPool);
				m_runtimeValues.push_front(RuntimeValue());
				auto leftNodeBlock = m_memPool->Alloc<RuntimeBlock>(binaryOp->m_arg0, m_bindingParent, GetHandle(), m_depTracker,
					&m_runtimeValues.front(), m_memPool);
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
			auto lArg = m_runtimeValues.front();
			m_runtimeValues.pop_front();
			auto rArg = m_runtimeValues.front();
			m_runtimeValues.pop_front();
			if (lArg.m_type != RuntimeValue::Type::Double || rArg.m_type != RuntimeValue::Type::Double)
			{
				throw std::invalid_argument("Expected both operands to add to be double");
			}
			RuntimeValue::Data dataVal;
			dataVal.doubleVal = BinaryOpFunctor()(lArg.m_data.doubleVal, rArg.m_data.doubleVal);
			FillDestValue(RuntimeValue::Type::Double, dataVal);
		}

		void AddDependentActiveBlock(const RuntimeBlockHandle_t block, const bool isNewDependency = true)
		{
			m_depTracker->AddActiveBlock(block);
			auto derefdBlock = m_memPool->derefHandle(block);
			if (derefdBlock->m_parent != RuntimeBlockHandle_t() && isNewDependency)
			{
				auto derefdParent = m_memPool->derefHandle(derefdBlock->m_parent);
				++derefdParent->m_dependenciesRemaining;
			}
		}

		void FillDestValue(const typename RuntimeValue::Type type, const typename RuntimeValue::Data& data)
		{
			m_dest->SetValue(type, data);
			if (m_parent != RuntimeBlockHandle_t())
			{
				auto derefdParent = m_memPool->derefHandle(m_parent);
				if (--derefdParent->m_dependenciesRemaining == 0)
				{
					m_depTracker->AddActiveBlock(m_parent);
				}
			}
		}

		Compiler::ASTNodeHandle m_astNode;
		List<RuntimeValue> m_runtimeValues;
		RuntimeBlockHandle_t m_bindingParent;
		RuntimeBlockHandle_t m_parent;
		DependencyTracker_t* m_depTracker;
		RuntimeValue* m_dest;
		std::atomic<int> m_dependenciesRemaining = 0;
		std::shared_ptr<PortableMemPool> m_memPool;
	};
}