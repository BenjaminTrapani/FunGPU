#pragma once

#include "Types.h"
#include <memory>
#include <string>
#include <vector>

namespace FunGPU {
class SExpr {
public:
  enum class Type { Symbol, Number, ListOfSExpr };
  SExpr();
  SExpr(const std::shared_ptr<std::string> &literalValue);
  void AddChild(const std::shared_ptr<SExpr> &val);
  void DebugPrint(Index_t indentLevel);
  Type GetType() const { return m_type; }
  const std::shared_ptr<std::string> GetSymbol() const { return m_stringValue; }
  const std::shared_ptr<std::vector<std::shared_ptr<SExpr>>>
  GetChildren() const {
    return m_sexprValue;
  }
  Float_t GetfloatVal() const { return m_numValue; }

private:
  Type m_type;
  std::shared_ptr<std::string> m_stringValue;
  Float_t m_numValue;
  std::shared_ptr<std::vector<std::shared_ptr<SExpr>>> m_sexprValue;
};
} // namespace FunGPU
