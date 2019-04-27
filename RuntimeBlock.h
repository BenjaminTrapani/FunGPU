#include "Compiler.h"
#include "List.hpp"
#include <atomic>

namespace FunGPU
{
	template <class DependencyTracker_t>
	class RuntimeBlock
	{
	public:
		class FunctionValue
		{
		public:
			FunctionValue() : m_expr(nullptr), m_bindingParent(nullptr), m_argCount(0) {}
			FunctionValue(Compiler::ASTNode* expr, RuntimeBlock* bindingParent, const Index_t argCount) : 
				m_expr(expr), m_bindingParent(bindingParent), m_argCount(argCount) {}
			bool operator==(const FunctionValue& other) const
			{
				return m_expr == other.m_expr && m_bindingParent == other.m_bindingParent;
			}

			Compiler::ASTNode* m_expr;
			RuntimeBlock* m_bindingParent;
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

		RuntimeBlock(Compiler::ASTNode* astNode, RuntimeBlock* bindingParent,
			RuntimeBlock* parent, DependencyTracker_t* depTracker, RuntimeValue* dest) :
			m_astNode(astNode), m_bindingParent(bindingParent), m_parent(parent),
			m_depTracker(depTracker), m_dest(dest) {
		}

		Compiler::ASTNode* GetASTNode()
		{
			return m_astNode;
		}

		void PerformEvalPass()
		{
			switch (m_astNode->m_type)
			{
			case Compiler::ASTNode::Type::Bind:
			case Compiler::ASTNode::Type::BindRec:
			{
				const bool isRec = m_astNode->m_type == Compiler::ASTNode::Type::BindRec;
				auto bindNode = static_cast<Compiler::BindNode*>(m_astNode);
				if (m_runtimeValues.size() == 0)
				{
					for (Index_t i = 0; i < bindNode->m_bindings.size(); ++i)
					{
						m_runtimeValues.push_front(RuntimeValue());
						auto targetRuntimeValue = &m_runtimeValues.front();
						auto dependencyOnBinding = new RuntimeBlock(bindNode->m_bindings.Get(i),
							isRec ? this : m_bindingParent, this, m_depTracker, targetRuntimeValue);
						AddDependentActiveBlock(dependencyOnBinding);
					}
				}
				else
				{
					auto depOnExpr = new RuntimeBlock(bindNode->m_childExpr, this, m_parent, m_depTracker, m_dest);
					AddDependentActiveBlock(depOnExpr, false);
				}

				break;
			}
			case Compiler::ASTNode::Type::Call:
			{
				auto callNode = static_cast<Compiler::CallNode*>(m_astNode);
				if (m_runtimeValues.size() == 0)
				{
					for (Index_t i = 0; i < callNode->m_args.size(); ++i)
					{
						m_runtimeValues.push_front(RuntimeValue());
						auto targetRuntimeValue = &m_runtimeValues.front();
						auto dependencyOnArg = new RuntimeBlock(callNode->m_args.Get(i), m_bindingParent, this,
							m_depTracker, targetRuntimeValue);
						AddDependentActiveBlock(dependencyOnArg);
					}

					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyOnLambda = new RuntimeBlock(callNode->m_target, m_bindingParent, this,
						m_depTracker, &m_runtimeValues.front());
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
					auto lambdaBlock = new RuntimeBlock(lambdaVal.m_data.functionVal.m_expr,
						this, m_parent, m_depTracker, m_dest);
					m_bindingParent = lambdaVal.m_data.functionVal.m_bindingParent;
					AddDependentActiveBlock(lambdaBlock, false);
				}

				break;
			}
			case Compiler::ASTNode::Type::If:
			{
				auto ifNode = static_cast<Compiler::IfNode*>(m_astNode);
				if (m_runtimeValues.size() == 0)
				{
					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyOnPred = new RuntimeBlock(ifNode->m_pred, m_bindingParent, this, m_depTracker, &m_runtimeValues.front());
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
					Compiler::ASTNode* branchToTake = isPredTrue ? ifNode->m_then : ifNode->m_else;
					auto dependencyOnBranch = new RuntimeBlock(branchToTake, m_bindingParent, m_parent, m_depTracker, m_dest);
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
				auto unaryOp = static_cast<Compiler::UnaryOpNode*>(m_astNode);
				if (m_runtimeValues.size() == 0)
				{
					m_runtimeValues.push_front(RuntimeValue());
					auto dependencyNode = new RuntimeBlock(unaryOp->m_arg0, m_bindingParent, this, m_depTracker, &m_runtimeValues.front());
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
				auto numNode = static_cast<Compiler::NumberNode*>(m_astNode);
				RuntimeValue::Data data;
				data.doubleVal = numNode->m_value;
				FillDestValue(RuntimeValue::Type::Double, data);
				break;
			}
			case Compiler::ASTNode::Type::Identifier:
			{
				auto identNode = static_cast<Compiler::IdentifierNode*>(m_astNode);
				auto identVal = GetRuntimeValueForIndex(identNode->m_index);
				FillDestValue(identVal->m_type, identVal->m_data);
				break;
			}
			case Compiler::ASTNode::Type::Lambda:
			{
				auto lambdaNode = static_cast<Compiler::LambdaNode*>(m_astNode);
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
			while (tempParent && index >= tempParent->m_runtimeValues.size())
			{
				index -= tempParent->m_runtimeValues.size();
				tempParent = tempParent->m_bindingParent;
			}
			if (tempParent == nullptr)
			{
				throw std::runtime_error("Failed to find runtime value for index");
			}

			// Index is in tempParent's runtime values. It is 'index' from beginning.
			return &tempParent->m_runtimeValues.GetItemAtIndex(index);
		}

		bool MaybeAddBinaryOp()
		{
			if (m_runtimeValues.size() == 0)
			{
				auto binaryOp = static_cast<Compiler::BinaryOpNode*>(m_astNode);
				m_runtimeValues.push_front(RuntimeValue());
				auto rightNodeBlock = new RuntimeBlock(binaryOp->m_arg1, m_bindingParent, this, m_depTracker, &m_runtimeValues.front());
				m_runtimeValues.push_front(RuntimeValue());
				auto leftNodeBlock = new RuntimeBlock(binaryOp->m_arg0, m_bindingParent, this, m_depTracker, &m_runtimeValues.front());
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

		void AddDependentActiveBlock(RuntimeBlock* block, const bool isNewDependency = true)
		{
			m_depTracker->AddActiveBlock(block);
			if (block->m_parent && isNewDependency)
			{
				++block->m_parent->m_dependenciesRemaining;
			}
		}

		void FillDestValue(const typename RuntimeValue::Type type, const typename RuntimeValue::Data& data)
		{
			m_dest->SetValue(type, data);
			if (m_parent)
			{
				if (--m_parent->m_dependenciesRemaining == 0)
				{
					m_depTracker->AddActiveBlock(m_parent);
				}
			}
		}

		Compiler::ASTNode* m_astNode;
		List<RuntimeValue> m_runtimeValues;
		RuntimeBlock* m_bindingParent = nullptr;
		RuntimeBlock* m_parent = nullptr;
		DependencyTracker_t* m_depTracker;
		RuntimeValue* m_dest;
		std::atomic<int> m_dependenciesRemaining = 0;
	};
}