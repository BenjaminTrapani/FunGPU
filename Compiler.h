#pragma once

#include "SExpr.h"
#include "Types.h"
#include "Array.hpp"

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
		
		class BindNode : public ASTNode
		{
		public:
			BindNode(const Index_t numBindings, const bool isRec) : ASTNode(isRec ? Type::BindRec : Type::Bind), 
				m_bindings(numBindings) {}
			Array<ASTNode*> m_bindings;
			ASTNode* m_childExpr;
		};

		class UnaryOpNode : public ASTNode
		{
		public:
			UnaryOpNode(ASTNode::Type type, ASTNode* arg0) : ASTNode(type), m_arg0(arg0) {}

			ASTNode* m_arg0;
		};

		class BinaryOpNode : public ASTNode
		{
		public:
			BinaryOpNode(ASTNode::Type type, ASTNode* arg0, ASTNode* arg1) : ASTNode(type), m_arg0(arg0), m_arg1(arg1) {}

			ASTNode* m_arg0;
			ASTNode* m_arg1;
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
			IfNode(ASTNode* pred, ASTNode* then, ASTNode* elseExpr) : ASTNode(ASTNode::Type::If),
				m_pred(pred), m_then(then), m_else(elseExpr) {}
			ASTNode* m_pred;
			ASTNode* m_then;
			ASTNode* m_else;
		};

		class LambdaNode : public ASTNode
		{
		public:
			LambdaNode(const Index_t argCount, ASTNode* childExpr) : 
				ASTNode(ASTNode::Type::Lambda), m_argCount(argCount), m_childExpr(childExpr) {}
			Index_t m_argCount;
			ASTNode* m_childExpr;
		};

		class CallNode : public ASTNode
		{
		public:
			CallNode(const Index_t argCount, ASTNode* target): ASTNode(ASTNode::Type::Call), 
				m_target(target), m_args(argCount) {}
			ASTNode* m_target;
			Array<ASTNode*> m_args;
		};

		class CompileException
		{
		public:
			CompileException(const std::string& what) : m_what(what) {}
			const std::string& What() const { return m_what; }
		private:
			std::string m_what;
		};

		Compiler(std::shared_ptr<const SExpr> sexpr) : m_sExpr(sexpr) {}

		ASTNode* Compile()
		{
			std::list<std::string> initialBound;
			return Compile(m_sExpr, initialBound);
		}

		void DebugPrintAST(ASTNode* rootOfAST);

	private:
		ASTNode* Compile(std::shared_ptr<const SExpr> sexpr, 
			std::list<std::string> boundIdentifiers);
		ASTNode* CompileListOfSExpr(std::shared_ptr<const SExpr> sexpr,
			std::list<std::string> boundIdentifiers);

		std::shared_ptr<const SExpr> m_sExpr;
	};
}