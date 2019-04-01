#pragma once

#include "SExpr.h"
#include <memory>
#include <fstream>

namespace FunGPU
{
	class Parser
	{
	public:
		Parser(const std::string& fileName);
		std::shared_ptr<SExpr> ParseProgram();
	private:
		struct ParsedExpr
		{
			virtual void DebugPrint() = 0;
			virtual ~ParsedExpr() {};
		};

		struct ParenthesizedExpr : public ParsedExpr
		{
			ParenthesizedExpr() : m_childExprs(std::make_shared<std::vector<std::shared_ptr<ParsedExpr>>>())
			{
			}

			void DebugPrint() override;

			std::shared_ptr<std::vector<std::shared_ptr<ParsedExpr>>> m_childExprs;
			size_t m_currentChildIndex = 0;
		};

		struct StringExpr : public ParsedExpr
		{
			void DebugPrint() override;

			std::string m_value;
		};

		std::shared_ptr<ParsedExpr> GetStringExprs();
		std::shared_ptr<SExpr> GetSExprFromStringExprs(const std::shared_ptr<ParsedExpr>& parenthesizedExprs);

		std::ifstream m_fileToRead;
	};
}