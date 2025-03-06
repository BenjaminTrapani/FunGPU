#include "Core/Compiler.hpp"
#include "Core/Error.hpp"
#include "Core/PortableMemPool.hpp"
#include "Core/RuntimeBlock.hpp"
#include <CL/sycl.hpp>
#include <array>
#include <memory>

namespace FunGPU {
class CPUEvaluator {
public:
  class DependencyTracker;

  using RuntimeBlock_t = RuntimeBlock<DependencyTracker, 8192 * 60>;
  using GarbageCollector_t = RuntimeBlock_t::GarbageCollector_t;

  class DependencyTracker {
    friend class CPUEvaluator;

  public:
    Error
    add_active_block(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block);
    void insert_active_block(const RuntimeBlock_t::SharedRuntimeBlockHandle_t &,
                             Index_t);

    Index_t get_active_block_count() {
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          active_block_count(m_active_block_count_data);
      return active_block_count.load();
    }

    void flip_active_blocks_buffer(const Index_t new_active_block_count = 0) {
      m_active_block_count_data = new_active_block_count;
      m_active_blocks_buffer_idx =
          (m_active_blocks_buffer_idx + 1) % m_active_blocks.size();
      m_prev_active_blocks_buffer_idx =
          (m_prev_active_blocks_buffer_idx + 1) % m_active_blocks.size();
    }

    RuntimeBlock_t::SharedRuntimeBlockHandle_t
    get_block_at_index(const Index_t index) {
      return m_active_blocks[m_prev_active_blocks_buffer_idx][index];
    }

  private:
    std::array<
        std::array<RuntimeBlock_t::SharedRuntimeBlockHandle_t, 8192 * 16>, 2>
        m_active_blocks;
    Index_t m_active_block_count_data = 0;
    Index_t m_active_blocks_buffer_idx = 0;
    Index_t m_prev_active_blocks_buffer_idx = 1;
  };

  CPUEvaluator(cl::sycl::buffer<PortableMemPool> mem_pool);
  ~CPUEvaluator();
  RuntimeBlock_t::RuntimeValue
  evaluate_program(const Compiler::ASTNodeHandle &root_node,
                   Index_t &max_concurrent_blocks);
  cl::sycl::buffer<PortableMemPool> get_mem_pool_buffer() const {
    return m_mem_pool_buff;
  }

private:
  void create_first_block(Compiler::ASTNodeHandle root_node);
  void check_for_block_errors(Index_t max_concurrent_blocks);
  void perform_garbage_collection(Index_t num_active_blocks);

  std::shared_ptr<DependencyTracker> m_dependency_tracker;
  std::shared_ptr<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>
      m_result_value;

  cl::sycl::buffer<PortableMemPool::Handle<GarbageCollector_t>>
      m_garbage_collector_handle_buff;
  cl::sycl::buffer<PortableMemPool> m_mem_pool_buff;
  cl::sycl::buffer<DependencyTracker> m_dependency_tracker_buff;
  cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>
      m_result_value_buff;
  cl::sycl::buffer<Error> m_errors_per_block;
  cl::sycl::buffer<Index_t> m_block_error_idx;
  cl::sycl::buffer<bool> m_markings_expanded;
  cl::sycl::buffer<Index_t> m_managed_allocd_count;
  cl::sycl::buffer<bool> m_requires_garbage_collection;
  cl::sycl::buffer<Index_t> m_num_active_blocks_buff;
  cl::sycl::buffer<RuntimeBlock_t::RuntimeValue> m_result_buffer_on_host;
  cl::sycl::queue m_work_queue;
};
} // namespace FunGPU
