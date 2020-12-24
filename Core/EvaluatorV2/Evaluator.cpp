#include "Core/Evaluator.hpp"

namespace FunGPU::EvaluatorV2 {
RuntimeBlockHandle
Evaluator::DependencyAggregator::runtime_block(const Index_t index) {
  return buffers[cur_buffer_idx_][index];
}

void Evaluator::DependencyAggregator::add_block(
    const RuntimeBlockHandle handle) {
  cl::sycl::atomic<int> atomic_num_active_blocks(
      (cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(
          num_active_blocks_)));
  const auto reserved_idx = atomic_num_active_blocks.fetch_add(1);
  buffers_[cur_buffer_idx_ & 1][reserved_idx] = handle;
}

Index_t Evaluator::DependencyAggregator::flip() {
  ++cur_buffer_idx_;
  const auto prev_num_active_blocks = num_active_blocks_;
  num_active_blocks_ = 0;
  return prev_num_active_blocks;
}

Evaluator::Evaluator(cl::sycl::buffer<PortableMemPool> buffer)
    : mem_pool_buffer_(buffer) {}

auto Evaluator::construct_initial_block(const Program &program)
    -> RuntimeBlockHandle {
  auto mem_pool_acc =
      mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>();
  const auto &main_lambda = mem_pool_acc[0].derefHandle(program)[0];
  auto main_block_handle = mem_pool_acc[0].alloc<RuntimeBlock_t>(main_lambda);
  return mem_pool_acc[0].derefHandle(main_block_handle);
}

RuntimeValue Evaluator::read_result(const RuntimeBlockHandle first_block,
                                    const Program &program) {
  auto mem_pool_acc =
      mem_pool_buffer_.get_access<cl::sycl::access::mode::read_write>();
  const auto &main_lambda = mem_pool_acc[0].derefHandle(program)[0];
  const auto &block = mem_pool_acc[0].derefHandle(first_block);
  return block.result(0, main_lambda);
}

RuntimeValue Evaluator::compute(const Program &program) {
  auto first_block = construct_initial_block(program);

  while (true) {
    // break if first_block becomes ready again and is complete
  }

  return read_result(first_block, program);
}
} // namespace FunGPU::EvaluatorV2
