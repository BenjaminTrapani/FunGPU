#pragma once

#include "core/ast_node.hpp"
#include "core/evaluator_v2/program.hpp"
#include "core/portable_mem_pool.hpp"
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace FunGPU::EvaluatorV2 {
class BlockGenerator {
public:
  BlockGenerator(cl::sycl::buffer<PortableMemPool>,
                 Index_t registers_per_block);
  Program construct_blocks(ASTNodeHandle root);

private:
  void assign_lambdas_block_indices(
      PortableMemPool::HostAccessor_t &mem_pool_acc, const ASTNodeHandle root,
      std::map<ASTNodeHandle, Index_t> &result, Index_t &cur_index);

  void compute_lambda_space_ident_to_use_count(
      const ASTNodeHandle &node, Index_t num_bound_so_far,
      PortableMemPool::HostAccessor_t &mem_pool_acc,
      std::unordered_map<Index_t, Index_t>
          &lambda_space_ident_to_remaining_count);

  void extract_lambda_space_captured_indices(
      const ASTNodeHandle &node, Index_t num_bound_so_far,
      PortableMemPool::HostAccessor_t &mem_pool_acc,
      std::set<Index_t> &captured_indices);

  Lambda
  construct_block(const ASTNodeHandle &lambda, const ASTNodeHandle root,
                  const std::map<ASTNodeHandle, Index_t> &lambdas_to_blocks,
                  PortableMemPool::HostAccessor_t &mem_pool_acc);

  cl::sycl::buffer<PortableMemPool> pool_;
  const Index_t registers_per_block_;
};
} // namespace FunGPU::EvaluatorV2
