#include "core/s_expr.hpp"
#include <cstdlib>
#include <iostream>

namespace FunGPU {
SExpr::SExpr()
    : m_type(Type::ListOfSExpr), m_float_val(0),
      m_children(
          std::make_shared<std::vector<std::shared_ptr<const SExpr>>>()) {}

SExpr::SExpr(const std::string &symbol) {
  try {
    const Float_t temp_float_val =
        static_cast<Float_t>(std::stof(symbol.c_str()));
    m_type = Type::Number;
    m_float_val = temp_float_val;
  } catch (const std::exception &) {
    m_type = Type::Symbol;
    m_symbol = std::make_shared<std::string>(symbol);
  }
}

void SExpr::add_child(const std::shared_ptr<const SExpr> &child) {
  m_children->push_back(child);
}

void SExpr::debug_print(const std::size_t indent) const {
  const auto print_indent = [&] {
    for (std::size_t i = 0; i < indent; ++i) {
      std::cout << " ";
    }
  };

  print_indent();
  switch (m_type) {
  case Type::Number:
    std::cout << m_float_val << std::endl;
    break;
  case Type::Symbol:
    std::cout << *m_symbol << std::endl;
    break;
  case Type::ListOfSExpr:
    std::cout << "(" << std::endl;
    for (const auto &child : *m_children) {
      child->debug_print(indent + 1);
    }
    print_indent();
    std::cout << ")" << std::endl;
    break;
  }
}
} // namespace FunGPU
