#pragma once

#include "core/types.hpp"
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

  Type get_type() const noexcept { return m_type; }
  Float_t get_float_val() const noexcept { return m_float_val; }
  const std::string &get_symbol() const noexcept { return m_symbol; }
  const std::vector<std::shared_ptr<const SExpr>> &
  get_children() const noexcept {
    return m_children;
  }

  void add_child(const std::shared_ptr<const SExpr> &child);
  void debug_print(const std::size_t indent) const;

private:
  Type m_type;
  Float_t m_float_val;
  std::string m_symbol;
  std::vector<std::shared_ptr<const SExpr>> m_children;
};
} // namespace FunGPU
