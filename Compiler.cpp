#include "Compiler.h"
#include <sstream>
#include <iostream>

namespace FunGPU
{
	Compiler::ASTNodeHandle Compiler::CompileListOfSExpr(std::shared_ptr<const SExpr> sexpr, std::list<std::string> boundIdentifiers)
	{
		auto sexprChildren = sexpr->GetChildren();
		if (sexprChildren->size() < 1)
		{
			throw CompileException("List of sexpr is less than 1, invalid expr");
		}

		auto firstChild = sexprChildren->at(0);
		if (firstChild->GetType() != SExpr::Type::Symbol)
		{
			throw CompileException("First child in sexpr list is not a symbol, invalid expr");
		}

		const auto firstChildSym = *firstChild->GetSymbol();
		ASTNodeHandle result;
		if (firstChildSym == "+")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected + to have 2 args");
			}
			result = m_memPool->Alloc<BinaryOpNode>(ASTNode::Type::Add, Compile(sexprChildren->at(1), boundIdentifiers),
				Compile(sexprChildren->at(2), boundIdentifiers));
		}
		else if (firstChildSym == "-")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected - to have 2 args");
			}
			result = m_memPool->Alloc<BinaryOpNode>(ASTNode::Type::Sub, Compile(sexprChildren->at(1), boundIdentifiers),
				Compile(sexprChildren->at(2), boundIdentifiers));
		}
		else if (firstChildSym == "*")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected * to have 2 args");
			}
			result = m_memPool->Alloc<BinaryOpNode>(ASTNode::Type::Mul, Compile(sexprChildren->at(1), boundIdentifiers),
				Compile(sexprChildren->at(2), boundIdentifiers));
		}
		else if (firstChildSym == "/")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected / to have 2 args");
			}
			result = m_memPool->Alloc<BinaryOpNode>(ASTNode::Type::Div, Compile(sexprChildren->at(1), boundIdentifiers),
				Compile(sexprChildren->at(2), boundIdentifiers));
		}
		else if (firstChildSym == "=" || firstChildSym == "eq?")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected = to have 2 args");
			}
			result = m_memPool->Alloc<BinaryOpNode>(ASTNode::Type::Equal, Compile(sexprChildren->at(1), boundIdentifiers),
				Compile(sexprChildren->at(2), boundIdentifiers));
		}
		else if (firstChildSym == ">")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected > to have 2 args");
			}
			result = m_memPool->Alloc<BinaryOpNode>(ASTNode::Type::GreaterThan, Compile(sexprChildren->at(1), boundIdentifiers),
				Compile(sexprChildren->at(2), boundIdentifiers));
		}
		else if (firstChildSym == "let" || firstChildSym == "letrec")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected let to have 2 args");
			}
			const bool isRec = firstChildSym == "letrec";

			auto bindingExprs = sexprChildren->at(1)->GetChildren();
			auto exprToEvalInBindingEnv = sexprChildren->at(2);
			const auto bindNodeHandle = m_memPool->Alloc<BindNode>(bindingExprs->size(), isRec);
			auto bindNode = m_memPool->derefHandle(bindNodeHandle);

			auto updatedBindings = boundIdentifiers;
			for (const auto& bindExpr : *bindingExprs)
			{
				auto bindExprChildren = bindExpr->GetChildren();
				if (bindExprChildren->size() != 2)
				{
					throw CompileException("Expected binding expr to have 2 elements (identifier, value expr)");
				}
				auto identExpr = bindExprChildren->at(0);
				if (identExpr->GetType() != SExpr::Type::Symbol)
				{
					throw CompileException("Expected first component of bind expr to be an identifier");
				}

				const auto identString = identExpr->GetSymbol();
				updatedBindings.push_front(*identString);
			}

			for (size_t i = 0; i < bindingExprs->size(); ++i)
			{
				auto bindExpr = bindingExprs->at(i);
				auto bindExprChildren = bindExpr->GetChildren();
				bindNode->m_bindings.Set(i, Compile(bindExprChildren->at(1), isRec ? updatedBindings : boundIdentifiers));
			}

			bindNode->m_childExpr = Compile(sexprChildren->at(2), updatedBindings);
			result = bindNodeHandle;
		}
		else if (firstChildSym == "lambda")
		{
			if (sexprChildren->size() != 3)
			{
				throw CompileException("Expected lambda to have 2 child exprs");
			}
			auto identifierList = sexprChildren->at(1);
			if (identifierList->GetType() != SExpr::Type::ListOfSExpr)
			{
				throw CompileException("Lambda arg list expected to be list of sexpr");
			}
			auto identifierListChildren = identifierList->GetChildren();
			for (auto identifier : *identifierListChildren)
			{
				if (identifier->GetType() != SExpr::Type::Symbol)
				{
					throw CompileException("Expected arguments in lambda expression to be symbols");
				}
				boundIdentifiers.push_front(*identifier->GetSymbol());
			}
			auto exprToEval = sexprChildren->at(2);
			auto compiledASTNode = Compile(exprToEval, boundIdentifiers);
			result = m_memPool->Alloc<LambdaNode>(identifierListChildren->size(), compiledASTNode);
		}
		else if (firstChildSym == "if")
		{
			if (sexprChildren->size() != 4)
			{
				throw CompileException("Expected if expr to have 3 arguments");
			}
			auto predChild = sexprChildren->at(1);
			auto thenChild = sexprChildren->at(2);
			auto elseChild = sexprChildren->at(3);
			result = m_memPool->Alloc<IfNode>(Compile(predChild, boundIdentifiers), Compile(thenChild, boundIdentifiers),
				Compile(elseChild, boundIdentifiers));
		}
		else if (firstChildSym == "floor")
		{
			if (sexprChildren->size() != 2)
			{
				throw CompileException("Expected floor to get 1 argument");
			}
			result = m_memPool->Alloc<UnaryOpNode>(ASTNode::Type::Floor,
				Compile(sexprChildren->at(1), boundIdentifiers));
		}
		else // This is hopefully a call to user-defined function.
		{
			auto argCount = sexprChildren->size() - 1;
			auto targetLambdaExpr = sexprChildren->at(0);
			const auto callNodeHandle = m_memPool->Alloc<CallNode>(argCount, Compile(targetLambdaExpr, boundIdentifiers));
			auto callNode = m_memPool->derefHandle(callNodeHandle);
			for (size_t i = 1; i < sexprChildren->size(); ++i)
			{
				auto curArg = sexprChildren->at(i);
				callNode->m_args.Set(i - 1, Compile(curArg, boundIdentifiers));
			}
			result = callNodeHandle;
		}

		if (result == ASTNodeHandle())
		{
			throw CompileException("Failed to compile list of sexpr");
		}

		return result;
	}

	Compiler::ASTNodeHandle Compiler::Compile(std::shared_ptr<const SExpr> sexpr,
		std::list<std::string> boundIdentifiers)
	{
		ASTNodeHandle result;
		switch (sexpr->GetType())
		{
		case SExpr::Type::Symbol:
		{
			auto identPos = std::find(boundIdentifiers.begin(), boundIdentifiers.end(), *sexpr->GetSymbol());
			if (identPos == boundIdentifiers.end())
			{
				std::stringstream sstream;
				sstream << "Unbound identifier " << *sexpr->GetSymbol() << std::endl;
				throw CompileException(sstream.str());
			}
			result = m_memPool->Alloc<IdentifierNode>(std::distance(boundIdentifiers.begin(), identPos));
			break;
		}
		case SExpr::Type::Number:
		{
			result = m_memPool->Alloc<NumberNode>(sexpr->GetDoubleVal());
			break;
		}
		case SExpr::Type::ListOfSExpr:
		{
			result = CompileListOfSExpr(sexpr, boundIdentifiers);
			break;
		}
		default:
			throw CompileException("Unexpected sexpr type");
			break;
		}

		if (result == ASTNodeHandle())
		{
			throw CompileException("Failed to compile sexpr");
		}

		return result;
	}

	void Compiler::DebugPrintAST(ASTNodeHandle rootOfASTHandle)
	{
		auto rootOfAST = m_memPool->derefHandle(rootOfASTHandle);
		switch (rootOfAST->m_type)
		{
		case ASTNode::Type::Bind:
		case ASTNode::Type::BindRec:
		{
			auto bindNode = static_cast<BindNode*>(rootOfAST);
			if (rootOfAST->m_type == ASTNode::Type::Bind)
			{
				std::cout << "(let ";
			}
			else
			{
				std::cout << "(letrec ";
			}
			std::cout << "(";
			for (Index_t i = 0; i < bindNode->m_bindings.size(); ++i)
			{
				DebugPrintAST(bindNode->m_bindings.Get(i));
				std::cout << " ";
			}
			std::cout << ")";
			std::cout << std::endl;
			DebugPrintAST(bindNode->m_childExpr);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::If:
		{
			std::cout << "(if ";
			auto ifNode = static_cast<IfNode*>(rootOfAST);
			DebugPrintAST(ifNode->m_pred);
			std::cout << std::endl;
			DebugPrintAST(ifNode->m_then);
			std::cout << std::endl;
			DebugPrintAST(ifNode->m_else);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Add:
		{
			std::cout << "(+ ";
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DebugPrintAST(binaryOpNode->m_arg0);
			DebugPrintAST(binaryOpNode->m_arg1);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Sub:
		{
			std::cout << "(- ";
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DebugPrintAST(binaryOpNode->m_arg0);
			DebugPrintAST(binaryOpNode->m_arg1);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Mul:
		{
			std::cout << "(* ";
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DebugPrintAST(binaryOpNode->m_arg0);
			DebugPrintAST(binaryOpNode->m_arg1);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Div:
		{
			std::cout << "(/ ";
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DebugPrintAST(binaryOpNode->m_arg0);
			DebugPrintAST(binaryOpNode->m_arg1);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Equal:
		{
			std::cout << "(= ";
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DebugPrintAST(binaryOpNode->m_arg0);
			DebugPrintAST(binaryOpNode->m_arg1);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::GreaterThan:
		{
			std::cout << "(> ";
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DebugPrintAST(binaryOpNode->m_arg0);
			DebugPrintAST(binaryOpNode->m_arg1);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Floor:
		{
			std::cout << "(floor ";
			auto unaryOpNode = static_cast<UnaryOpNode*>(rootOfAST);
			DebugPrintAST(unaryOpNode->m_arg0);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Number:
		{
			auto numNode = static_cast<NumberNode*>(rootOfAST);
			std::cout << numNode->m_value;
			break;
		}
		case ASTNode::Type::Identifier:
		{
			auto identNode = static_cast<IdentifierNode*>(rootOfAST);
			std::cout << "(ident: " << identNode->m_index << ")";
			break;
		}
		case ASTNode::Type::Lambda:
		{
			auto lambdaNode = static_cast<LambdaNode*>(rootOfAST);
			std::cout << "(lambda (argCount: " << lambdaNode->m_argCount << ")";
			std::cout << std::endl;
			DebugPrintAST(lambdaNode->m_childExpr);
			std::cout << ")";
			break;
		}
		case ASTNode::Type::Call:
		{
			auto callExpr = static_cast<CallNode*>(rootOfAST);
			std::cout << "(call ";
			DebugPrintAST(callExpr->m_target);
			std::cout << " ";
			for (Index_t i = 0; i < callExpr->m_args.size(); ++i)
			{
				DebugPrintAST(callExpr->m_args.Get(i));
				std::cout << " ";
			}
			std::cout << ")";
			break;
		}
		default:
			throw CompileException("Unexpected AST node type during debug print");
		}
	}

	void Compiler::DeallocateAST(const ASTNodeHandle rootOfASTHandle)
	{
		auto rootOfAST = m_memPool->derefHandle(rootOfASTHandle);
		switch (rootOfAST->m_type)
		{
		case ASTNode::Type::Bind:
		case ASTNode::Type::BindRec:
		{
			auto bindNode = static_cast<BindNode*>(rootOfAST);
			for (Index_t i = 0; i < bindNode->m_bindings.size(); ++i)
			{
				DeallocateAST(bindNode->m_bindings.Get(i));
			}
			DeallocateAST(bindNode->m_childExpr);
			break;
		}
		case ASTNode::Type::If:
		{
			auto ifNode = static_cast<IfNode*>(rootOfAST);
			DeallocateAST(ifNode->m_pred);
			DeallocateAST(ifNode->m_then);
			DeallocateAST(ifNode->m_else);
			break;
		}
		case ASTNode::Type::Add:
		{
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DeallocateAST(binaryOpNode->m_arg0);
			DeallocateAST(binaryOpNode->m_arg1);
			break;
		}
		case ASTNode::Type::Sub:
		{
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DeallocateAST(binaryOpNode->m_arg0);
			DeallocateAST(binaryOpNode->m_arg1);
			break;
		}
		case ASTNode::Type::Mul:
		{
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DeallocateAST(binaryOpNode->m_arg0);
			DeallocateAST(binaryOpNode->m_arg1);
			break;
		}
		case ASTNode::Type::Div:
		{
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DeallocateAST(binaryOpNode->m_arg0);
			DeallocateAST(binaryOpNode->m_arg1);
			break;
		}
		case ASTNode::Type::Equal:
		{
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DeallocateAST(binaryOpNode->m_arg0);
			DeallocateAST(binaryOpNode->m_arg1);
			break;
		}
		case ASTNode::Type::GreaterThan:
		{
			auto binaryOpNode = static_cast<BinaryOpNode*>(rootOfAST);
			DeallocateAST(binaryOpNode->m_arg0);
			DeallocateAST(binaryOpNode->m_arg1);
			break;
		}
		case ASTNode::Type::Floor:
		{
			auto unaryOpNode = static_cast<UnaryOpNode*>(rootOfAST);
			DeallocateAST(unaryOpNode->m_arg0);
			break;
		}
		case ASTNode::Type::Number:
			break;
		case ASTNode::Type::Identifier:
			break;
		case ASTNode::Type::Lambda:
		{
			auto lambdaNode = static_cast<LambdaNode*>(rootOfAST);
			DeallocateAST(lambdaNode->m_childExpr);
			break;
		}
		case ASTNode::Type::Call:
		{
			auto callExpr = static_cast<CallNode*>(rootOfAST);
			DeallocateAST(callExpr->m_target);
			for (Index_t i = 0; i < callExpr->m_args.size(); ++i)
			{
				DeallocateAST(callExpr->m_args.Get(i));
			}
			break;
		}
		default:
			throw CompileException("Unexpected AST node type during debug print");
		}

		m_memPool->Dealloc(rootOfASTHandle);
	}
}
