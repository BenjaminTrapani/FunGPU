#include "SExpr.h"
#include <cstdlib>
#include <iostream>

namespace FunGPU
{
	SExpr::SExpr() :
		m_type(Type::ListOfSExpr), m_sexprValue(std::make_shared<std::vector<std::shared_ptr<SExpr>>>()) {}

	SExpr::SExpr(const std::shared_ptr<std::string>& literalValue)
	{
		try
		{
			const double tempVal = std::stod(literalValue->c_str());
			m_type = Type::Number;
			m_numValue = tempVal;
		}
		catch (const std::exception&)
		{
			m_type = Type::Symbol;
			m_stringValue = literalValue;
		}
	}

	void SExpr::AddChild(const std::shared_ptr<SExpr>& val)
	{
		m_sexprValue->push_back(val);
	}

	void SExpr::DebugPrint(const unsigned int indentLevel)
	{
		switch (m_type)
		{
		case Type::Symbol:
		{
			std::cout << " " << *m_stringValue << " ";
			break;
		}
		case Type::Number:
		{
			std::cout << " " << m_numValue << " ";
			break;
		}
		case Type::ListOfSExpr:
		{
			std::cout << std::endl;
			for (unsigned int i = 0; i < indentLevel; ++i)
			{
				std::cout << " ";
			}
			std::cout << "(";
			const unsigned int indentedTwo = indentLevel + 2;
			for (auto subExpr : *m_sexprValue)
			{
				subExpr->DebugPrint(indentedTwo);
			}
			
			std::cout << ")";
			break;
		}
		default:
			throw std::invalid_argument("Unexpected type in debug print");
		}
	}
}
