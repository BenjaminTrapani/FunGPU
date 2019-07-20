#pragma once

#include "SExpr.h"
#include "Types.h"
#include "Array.hpp"
#include "PortableMemPool.h"

#include <memory>
#include <list>

namespace FunGPU
{
	class Compiler
	{
	public:
		class ASTNode
		{
		public:
			enum class Type
			{
				Bind,
				BindRec,
				Call,
				If,
				Add,
				Sub,
				Mul,
				Div,
				Equal,
				GreaterThan,
				Floor,
				Number,
				Identifier,
				Lambda,
			};
			ASTNode(Type type) : m_type(type) {}
			Type m_type;
		};
		
		using ASTNodeHandle = PortableMemPool::Handle<ASTNode>;

		class BindNode : public ASTNode
		{
		public:
			BindNode(const Index_t numBindings, const bool isRec, PortableMemPool* pool) : ASTNode(isRec ? Type::BindRec : Type::Bind), 
				m_bindings(pool->AllocArray<ASTNodeHandle>(numBindings)) {}
			PortableMemPool::ArrayHandle<ASTNodeHandle> m_bindings;
			ASTNodeHandle m_childExpr;
		};

		class UnaryOpNode : public ASTNode
		{
		public:
			UnaryOpNode(ASTNode::Type type, ASTNodeHandle arg0) : ASTNode(type), m_arg0(arg0) {}

			ASTNodeHandle m_arg0;
		};

		class BinaryOpNode : public ASTNode
		{
		public:
			BinaryOpNode(ASTNode::Type type, ASTNodeHandle arg0, ASTNodeHandle arg1) : ASTNode(type), 
				m_arg0(arg0), m_arg1(arg1) {}

			ASTNodeHandle m_arg0;
			ASTNodeHandle m_arg1;
		};

		class NumberNode : public ASTNode
		{
		public:
			NumberNode(const Float_t value) : ASTNode(ASTNode::Type::Number), m_value(value) {}
			Float_t m_value;
		};

		class IdentifierNode : public ASTNode
		{
		public:
			IdentifierNode(const Index_t index) : ASTNode(ASTNode::Type::Identifier), m_index(index) {}
			Index_t m_index;
		};

		class IfNode : public ASTNode
		{
		public:
			IfNode(ASTNodeHandle pred, ASTNodeHandle then, ASTNodeHandle elseExpr) : ASTNode(ASTNode::Type::If),
				m_pred(pred), m_then(then), m_else(elseExpr) {}
			ASTNodeHandle m_pred;
			ASTNodeHandle m_then;
			ASTNodeHandle m_else;
		};

		class LambdaNode : public ASTNode
		{
		public:
			LambdaNode(const Index_t argCount, ASTNodeHandle childExpr) :
				ASTNode(ASTNode::Type::Lambda), m_argCount(argCount), m_childExpr(childExpr) {}
			Index_t m_argCount;
			ASTNodeHandle m_childExpr;
		};

		class CallNode : public ASTNode
		{
		public:
			CallNode(const Index_t argCount, ASTNodeHandle target, PortableMemPool* pool): ASTNode(ASTNode::Type::Call),
				m_target(target), m_args(pool->AllocArray<ASTNodeHandle>(argCount)) {}
			ASTNodeHandle m_target;
			PortableMemPool::ArrayHandle<ASTNodeHandle> m_args;
		};

		class CompileException
		{
		public:
			CompileException(const std::string& what) : m_what(what) {}
			const std::string& What() const { return m_what; }
		private:
			std::string m_what;
		};

		Compiler(std::shared_ptr<const SExpr> sexpr, const std::shared_ptr<PortableMemPool>& pool) : 
			m_sExpr(sexpr), m_memPool(pool) {}

		ASTNodeHandle Compile()
		{
			std::list<std::string> initialBound;
			return Compile(m_sExpr, initialBound);
		}

		void DebugPrintAST(ASTNodeHandle rootOfAST);
		void DeallocateAST(const ASTNodeHandle rootOfAST);

	private:
		ASTNodeHandle Compile(std::shared_ptr<const SExpr> sexpr,
			std::list<std::string> boundIdentifiers);
		ASTNodeHandle CompileListOfSExpr(std::shared_ptr<const SExpr> sexpr,
			std::list<std::string> boundIdentifiers);

		std::shared_ptr<const SExpr> m_sExpr;
		std::shared_ptr<PortableMemPool> m_memPool;
	};
}