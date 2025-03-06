#include "Core/Parser.hpp"
#include <iostream>
#include <stack>

namespace FunGPU {
void Parser::ParenthesizedExpr::debug_print() {
  std::cout << "(";
  for (const auto &child_expr : *m_child_exprs) {
    child_expr->debug_print();
    std::cout << " ";
  }
  std::cout << ")";
}

void Parser::StringExpr::debug_print() { std::cout << m_value; }

Parser::Parser(const std::string &file_name) : m_file_to_read(file_name) {
  if (!m_file_to_read.is_open()) {
    throw std::invalid_argument("Failed to open file " + file_name);
  }
}

std::shared_ptr<Parser::ParsedExpr> Parser::get_string_exprs() {
  auto root_of_result = std::make_shared<ParenthesizedExpr>();
  std::stack<std::shared_ptr<ParenthesizedExpr>> expr_stack;
  expr_stack.push(root_of_result);

  bool start_new_string = true;
  while (m_file_to_read.good()) {
    const char cur_char = m_file_to_read.get();
    auto &working_expr = expr_stack.top();
    switch (cur_char) {
    case '(': {
      auto sub_expr = std::make_shared<ParenthesizedExpr>();
      working_expr->m_child_exprs->push_back(sub_expr);
      expr_stack.push(sub_expr);
      start_new_string = true;
      break;
    }
    case ')':
      expr_stack.pop();
      break;
    case '\n':
    case '\t':
    case ' ':
    case '\r': {
      start_new_string = true;
      break;
    }
    case -1:
      break;
    default:
      if (start_new_string) {
        auto new_string = std::make_shared<StringExpr>();
        working_expr->m_child_exprs->push_back(new_string);
        start_new_string = false;
      }
      auto most_recent_child_expr = working_expr->m_child_exprs->at(
          working_expr->m_child_exprs->size() - 1);
      auto cur_string =
          std::dynamic_pointer_cast<StringExpr>(most_recent_child_expr);
      if (cur_string == nullptr) {
        throw std::invalid_argument("Expected to be appending to an "
                                    "in-progress string expr, but that was not "
                                    "the case");
      }
      cur_string->m_value += cur_char;
      break;
    }
  }

  if (root_of_result->m_child_exprs->size() != 1) {
    throw std::invalid_argument(
        "Expected exactly one top-level expression for input program");
  }

  return root_of_result->m_child_exprs->at(0);
}

std::shared_ptr<SExpr> Parser::get_sexpr_from_string_exprs(
    const std::shared_ptr<ParsedExpr> &parenthesized_exprs) {
  struct ParsedToSExprState {
    std::shared_ptr<SExpr> working_expr;
    std::shared_ptr<ParsedExpr> parsed_for_working;
  };

  std::stack<ParsedToSExprState> working_stack;
  ParsedToSExprState initial_state;
  initial_state.parsed_for_working = parenthesized_exprs;
  working_stack.push(initial_state);

  std::shared_ptr<SExpr> result_root;
  while (!working_stack.empty()) {
    auto &cur_state = working_stack.top();
    auto str_expr_here =
        std::dynamic_pointer_cast<StringExpr>(cur_state.parsed_for_working);
    if (str_expr_here) {
      auto new_sexpr = std::make_shared<SExpr>(str_expr_here->m_value);
      if (cur_state.working_expr) {
        cur_state.working_expr->add_child(new_sexpr);
      }

      if (result_root == nullptr) {
        result_root = new_sexpr;
      }
      working_stack.pop();
    } else {
      auto paren_expr = std::dynamic_pointer_cast<ParenthesizedExpr>(
          cur_state.parsed_for_working);
      std::shared_ptr<SExpr> sexpr_for_this_node;
      if (paren_expr->m_current_child_index == 0) {
        ParsedToSExprState state_for_this_node;
        state_for_this_node.parsed_for_working = cur_state.parsed_for_working;
        state_for_this_node.working_expr = std::make_shared<SExpr>();
        if (cur_state.working_expr) {
          cur_state.working_expr->add_child(state_for_this_node.working_expr);
        }
        if (!result_root) {
          result_root = state_for_this_node.working_expr;
        }
        sexpr_for_this_node = state_for_this_node.working_expr;
        working_stack.push(state_for_this_node);
      } else {
        sexpr_for_this_node = cur_state.working_expr;
      }

      if (paren_expr->m_current_child_index >=
          paren_expr->m_child_exprs->size()) {
        if (paren_expr->m_current_child_index == 0) {
          working_stack.pop();
        }
        working_stack.pop();
      } else {
        auto cur_child_parsed_expr =
            paren_expr->m_child_exprs->at(paren_expr->m_current_child_index);

        ParsedToSExprState child_state;
        child_state.parsed_for_working = cur_child_parsed_expr;
        child_state.working_expr = sexpr_for_this_node;
        working_stack.push(child_state);

        ++paren_expr->m_current_child_index;
      }
    }
  }

  return result_root;
}

std::shared_ptr<SExpr> Parser::parse_program() {
  const auto string_exprs = get_string_exprs();
  return get_sexpr_from_string_exprs(string_exprs);
}
} // namespace FunGPU
