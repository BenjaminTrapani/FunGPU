#pragma once

#include "core/evaluator_v2/program.hpp"
#include "core/compiler.hpp"
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
  Program construct_blocks(Compiler::ASTNodeHandle root);

private:
  void assign_lambdas_block_indices(
      PortableMemPool::HostAccessor_t &mem_pool_acc,
      const Compiler::ASTNodeHandle root,
      std::map<Compiler::ASTNodeHandle, Index_t> &result, Index_t &cur_index);

  void compute_lambda_space_ident_to_use_count(
      const Compiler::ASTNodeHandle &node, Index_t num_bound_so_far,
      PortableMemPool::HostAccessor_t &mem_pool_acc,
      std::unordered_map<Index_t, Index_t>
          &lambda_space_ident_to_remaining_count);

  void extract_lambda_space_captured_indices(
      const Compiler::ASTNodeHandle &node, Index_t num_bound_so_far,
      PortableMemPool::HostAccessor_t &mem_pool_acc,
      std::set<Index_t> &captured_indices);

  Lambda construct_block(
      const Compiler::ASTNodeHandle &lambda, const Compiler::ASTNodeHandle root,
      const std::map<Compiler::ASTNodeHandle, Index_t> &lambdas_to_blocks,
      PortableMemPool::HostAccessor_t &mem_pool_acc);

  cl::sycl::buffer<PortableMemPool> pool_;
  const Index_t registers_per_block_;
};
} // namespace FunGPU::EvaluatorV2
