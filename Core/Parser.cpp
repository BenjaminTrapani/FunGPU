#include "Parser.h"
#include <stack>
#include <iostream>

namespace FunGPU
{
	void Parser::ParenthesizedExpr::DebugPrint()
	{
		std::cout << "(";
		for (auto subExpr : *m_childExprs)
		{
			subExpr->DebugPrint();
		}
		std::cout << ")";
	}

	void Parser::StringExpr::DebugPrint()
	{
		std::cout << " " << m_value << " ";
	}
	
	Parser::Parser(const std::string& fileName) : m_fileToRead(fileName, std::ios::binary) {}
	
	std::shared_ptr<SExpr> Parser::ParseProgram()
	{
		auto stringExprs = GetStringExprs();
		return GetSExprFromStringExprs(stringExprs);
	}

	std::shared_ptr<Parser::ParsedExpr> Parser::GetStringExprs()
	{
		auto rootOfResult = std::make_shared<ParenthesizedExpr>();
		std::stack<std::shared_ptr<ParenthesizedExpr>> exprStack;
		exprStack.push(rootOfResult);

		bool startNewString = true;
		while (m_fileToRead.good())
		{
			const char curChar = m_fileToRead.get();
			auto& workingExpr = exprStack.top();
			switch (curChar)
			{
			case '(':
			{
				auto subExpr = std::make_shared<ParenthesizedExpr>();
				workingExpr->m_childExprs->push_back(subExpr);
				exprStack.push(subExpr);
				startNewString = true;
				break;
			}
			case ')':
				exprStack.pop();
				break;
			case '\n':
			case '\t':
			case ' ':
			case '\r':
			{
				startNewString = true;
				break;
			}
			case -1:
				break;
			default:
				if (startNewString)
				{
					auto newString = std::make_shared<StringExpr>();
					workingExpr->m_childExprs->push_back(newString);
					startNewString = false;
				}
				auto mostRecentChildExpr = workingExpr->m_childExprs->at(workingExpr->m_childExprs->size() - 1);
				auto curString = std::dynamic_pointer_cast<StringExpr>(mostRecentChildExpr);
				if (curString == nullptr)
				{
					throw std::invalid_argument("Expected to be appending to an in-progress string expr, but that was not the case");
				}
				curString->m_value += curChar;
				break;
			}
		}

		if (rootOfResult->m_childExprs->size() != 1)
		{
			throw std::invalid_argument("Expected exactly one top-level expression for input program");
		}

		return rootOfResult->m_childExprs->at(0);
	}

	std::shared_ptr<SExpr> Parser::GetSExprFromStringExprs(const std::shared_ptr<ParsedExpr>& parenthesizedExprs)
	{
		struct ParsedToSExprState {
			std::shared_ptr<SExpr> m_workingExpr;
			std::shared_ptr<ParsedExpr> m_parsedForWorking;
		};

		std::stack<ParsedToSExprState> workingStack;
		ParsedToSExprState initialState;
		initialState.m_parsedForWorking = parenthesizedExprs;
		workingStack.push(initialState);

		std::shared_ptr<SExpr> resultRoot;
		while (!workingStack.empty())
		{
			auto& curState = workingStack.top();
			auto strExprHere = std::dynamic_pointer_cast<StringExpr>(curState.m_parsedForWorking);
			if (strExprHere)
			{
				auto newSexpr = std::make_shared<SExpr>(std::make_shared<std::string>(strExprHere->m_value));
				if (curState.m_workingExpr)
				{
					curState.m_workingExpr->AddChild(newSexpr);
				}

				if (resultRoot == nullptr)
				{
					resultRoot = newSexpr;
				}
				workingStack.pop();
			}
			else
			{
				auto parenExpr = std::dynamic_pointer_cast<ParenthesizedExpr>(curState.m_parsedForWorking);
				std::shared_ptr<SExpr> sexprForThisNode;
				if (parenExpr->m_currentChildIndex == 0)
				{
					ParsedToSExprState stateForThisNode;
					stateForThisNode.m_parsedForWorking = curState.m_parsedForWorking;
					stateForThisNode.m_workingExpr = std::make_shared<SExpr>();
					if (curState.m_workingExpr)
					{
						curState.m_workingExpr->AddChild(stateForThisNode.m_workingExpr);
					}
					if (!resultRoot)
					{
						resultRoot = stateForThisNode.m_workingExpr;
					}
					sexprForThisNode = stateForThisNode.m_workingExpr;
					workingStack.push(stateForThisNode);
				}
				else
				{
					sexprForThisNode = curState.m_workingExpr;
				}

				if (parenExpr->m_currentChildIndex >= parenExpr->m_childExprs->size())
				{
					if (parenExpr->m_currentChildIndex == 0)
					{
						workingStack.pop();
					}
					workingStack.pop();
				}
				else
				{
					auto curChildParsedExpr = parenExpr->m_childExprs->at(parenExpr->m_currentChildIndex);
					
					ParsedToSExprState childState;
					childState.m_parsedForWorking = curChildParsedExpr;
					childState.m_workingExpr = sexprForThisNode;
					workingStack.push(childState);

					++parenExpr->m_currentChildIndex;
				}
			}
		}

		return resultRoot;
	}
}