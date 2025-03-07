#pragma once

#include "Core/s_expr.hpp"
#include "Core/types.hpp"
#include <fstream>
#include <memory>

namespace FunGPU {
class Parser {
public:
  Parser(const std::string &file_name);
  std::shared_ptr<SExpr> parse_program();

private:
  struct ParsedExpr {
    virtual void debug_print() = 0;
    virtual ~ParsedExpr() {};
  };

  struct ParenthesizedExpr : public ParsedExpr {
    ParenthesizedExpr()
        : m_child_exprs(
              std::make_shared<std::vector<std::shared_ptr<ParsedExpr>>>()) {}

    void debug_print() override;

    std::shared_ptr<std::vector<std::shared_ptr<ParsedExpr>>> m_child_exprs;
    Index_t m_current_child_index = 0;
  };

  struct StringExpr : public ParsedExpr {
    void debug_print() override;

    std::string m_value;
  };

  std::shared_ptr<ParsedExpr> get_string_exprs();
  std::shared_ptr<SExpr> get_sexpr_from_string_exprs(
      const std::shared_ptr<ParsedExpr> &parenthesized_exprs);

  std::ifstream m_file_to_read;
};
} // namespace FunGPU
