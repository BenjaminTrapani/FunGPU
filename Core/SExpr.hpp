#pragma once

#include "Core/Types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace FunGPU {
class SExpr {
public:
  enum class Type {
    Number,
    Symbol,
    ListOfSExpr,
  };

  SExpr();
  SExpr(const std::string &literalValue);

  Type get_type() const { return m_type; }
  Float_t get_float_val() const { return m_float_val; }
  std::shared_ptr<const std::string> get_symbol() const { return m_symbol; }
  std::shared_ptr<const std::vector<std::shared_ptr<const SExpr>>>
  get_children() const {
    return m_children;
  }

  void add_child(const std::shared_ptr<const SExpr> &child);
  void debug_print(const std::size_t indent) const;

private:
  Type m_type;
  Float_t m_float_val;
  std::shared_ptr<const std::string> m_symbol;
  std::shared_ptr<std::vector<std::shared_ptr<const SExpr>>> m_children;
};
} // namespace FunGPU
