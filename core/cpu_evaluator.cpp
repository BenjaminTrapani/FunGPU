#include "core/cpu_evaluator.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cl::sycl;

namespace FunGPU {
class create_gc;
class init_first_block;
class run_eval_pass;
class prep_blocks;
class check_requires_garbage_collection;
class get_managed_allocd_count;
class gc_initial_mark;
class gc_mark;
class gc_sweep;
class gc_prepare_compact;
class gc_compact;

Error CPUEvaluator::DependencyTracker::add_active_block(
    const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block) {
  cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                       cl::sycl::memory_scope::device,
                       cl::sycl::access::address_space::global_space>
      active_block_count(m_active_block_count_data);
  const auto index_to_insert = active_block_count.fetch_add(1U);
  auto &dest_array = m_active_blocks[m_active_blocks_buffer_idx];
  if (index_to_insert >= dest_array.size()) {
    return Error(Error::Type::EvaluatorOutOfActiveBlocks);
  }
  dest_array[index_to_insert] = block;

  return Error();
}

void CPUEvaluator::DependencyTracker::insert_active_block(
    const RuntimeBlock_t::SharedRuntimeBlockHandle_t &block,
    const Index_t index) {
  m_active_blocks[m_active_blocks_buffer_idx][index] = block;
}

CPUEvaluator::CPUEvaluator(cl::sycl::buffer<PortableMemPool> mem_pool)
    : m_dependency_tracker(std::make_shared<DependencyTracker>()),
      m_result_value(std::make_shared<
                     PortableMemPool::Handle<RuntimeBlock_t::RuntimeValue>>()),
      m_garbage_collector_handle_buff(range<1>(1)), m_mem_pool_buff(mem_pool),
      m_dependency_tracker_buff(m_dependency_tracker, range<1>(1)),
      m_result_value_buff(m_result_value, range<1>(1)),
      m_errors_per_block(
          range<1>(m_dependency_tracker->m_active_blocks[0].size())),
      m_block_error_idx(range<1>(1)), m_markings_expanded(range<1>(1)),
      m_managed_allocd_count(range<1>(1)),
      m_requires_garbage_collection(range<1>(1)),
      m_num_active_blocks_buff(range<1>(1)),
      m_result_buffer_on_host(range<1>(1))
/* m_work_queue(host_selector{})*/ {
  std::cout << std::endl;
  std::cout << "Running on "
            << m_work_queue.get_device().get_info<info::device::name>()
            << std::endl;
  std::cout << "Runtime block size in bytes: " << sizeof(RuntimeBlock_t)
            << std::endl;
  std::cout << "Runtime value size in bytes: "
            << sizeof(typename RuntimeBlock_t::RuntimeValue) << std::endl;
  try {
    m_work_queue.submit([&](handler &cgh) {
      auto mem_pool_write =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      auto garbage_collector_handle_acc =
          m_garbage_collector_handle_buff.get_access<access::mode::read_write>(
              cgh);
      auto mem_pool_host_acc =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      auto result_value_write =
          m_result_value_buff.get_access<access::mode::discard_write>(cgh);
      cgh.single_task<class create_gc>(
          [mem_pool_write, garbage_collector_handle_acc, result_value_write]() {
            garbage_collector_handle_acc[0] =
                mem_pool_write[0].alloc<GarbageCollector_t>();
            result_value_write[0] =
                mem_pool_write[0].alloc<RuntimeBlock_t::RuntimeValue>();
          });
    });
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception in init: " << e.what() << std::endl;
  }
}

CPUEvaluator::~CPUEvaluator() {
  m_work_queue.submit([&](handler &cgh) {
    auto mem_pool_acc =
        m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
    auto result_value_acc =
        m_result_value_buff.get_access<access::mode::read_write>(cgh);
    auto gc_handle_acc =
        m_garbage_collector_handle_buff.get_access<access::mode::read_write>(
            cgh);
    cgh.single_task<class evaluator_dealloc>(
        [mem_pool_acc, gc_handle_acc, result_value_acc]() {
          mem_pool_acc[0].dealloc(result_value_acc[0]);
          mem_pool_acc[0].dealloc(gc_handle_acc[0]);
        });
  });
}

void CPUEvaluator::create_first_block(const Compiler::ASTNodeHandle root_node) {
  try {
    cl::sycl::buffer<Error> alloc_gc_error_buf(cl::sycl::range<1>(1));
    m_work_queue.submit([&](handler &cgh) {
      auto mem_pool_write =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      auto dependency_tracker =
          m_dependency_tracker_buff.get_access<access::mode::read_write>(cgh);
      auto garbage_collector_handle_acc =
          m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);
      auto result_value_acc =
          m_result_value_buff.get_access<access::mode::read>(cgh);
      auto alloc_gc_error_acc =
          alloc_gc_error_buf.get_access<access::mode::discard_write>(cgh);
      cgh.single_task<class init_first_block>(
          [mem_pool_write, dependency_tracker, result_value_acc, root_node,
           garbage_collector_handle_acc, alloc_gc_error_acc]() {
            auto gc_ref =
                mem_pool_write[0].deref_handle(garbage_collector_handle_acc[0]);
            const RuntimeBlock_t::SharedRuntimeBlockHandle_t empty_block;
            RuntimeBlock_t::SharedRuntimeBlockHandle_t shared_initial_block;
            const auto alloc_error = gc_ref->alloc_managed(
                mem_pool_write, shared_initial_block, root_node, empty_block,
                empty_block, dependency_tracker, result_value_acc[0],
                mem_pool_write, garbage_collector_handle_acc[0]);
            alloc_gc_error_acc[0] = alloc_error;
            if (alloc_error.GetType() == Error::Type::Success) {
              alloc_gc_error_acc[0] =
                  dependency_tracker[0].add_active_block(shared_initial_block);
            }
          });
    });
    auto alloc_gc_error_val =
        alloc_gc_error_buf.get_access<access::mode::read>()[0];
    if (alloc_gc_error_val.GetType() != Error::Type::Success) {
      std::stringstream error_stream;
      error_stream << "Error allocating initial block: "
                   << static_cast<std::underlying_type_t<Error::Type>>(
                          alloc_gc_error_val.GetType())
                   << std::endl;
      throw std::runtime_error(error_stream.str());
    }
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }
  std::cout << "Successfully created first block" << std::endl;
}

void CPUEvaluator::check_for_block_errors(
    const Index_t max_concurrent_blocks_during_exec) {
  auto host_block_has_error_acc =
      m_block_error_idx.get_access<access::mode::read_write>();
  if (host_block_has_error_acc[0] > 0) {
    auto errors_per_block_acc =
        m_errors_per_block.get_access<access::mode::read_write>();
    std::stringstream ss;
    ss << "Block errors in previous pass: " << std::endl;
    for (Index_t i = 0; i < host_block_has_error_acc[0]; ++i) {
      ss << static_cast<int>(errors_per_block_acc[i].GetType()) << std::endl;
    }
    ss << "Max concurrent blocks during exec: "
       << max_concurrent_blocks_during_exec << std::endl;
    const auto error_string = ss.str();
    throw std::runtime_error(error_string);
  }
  host_block_has_error_acc[0] = 0;
}

void CPUEvaluator::perform_garbage_collection(const Index_t num_active_blocks) {
  if (num_active_blocks > 0) {
    m_work_queue.submit([&](handler &cgh) {
      auto mem_pool_acc =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      auto dependency_tracker_acc =
          m_dependency_tracker_buff.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class gc_initial_mark>(
          cl::sycl::range<1>(num_active_blocks),
          [mem_pool_acc, dependency_tracker_acc](item<1> itm) {
            const auto idx = itm.get_linear_id();
            const auto derefd_working = mem_pool_acc[0].deref_handle(
                dependency_tracker_acc[0].get_block_at_index(idx));
            derefd_working->set_marked();
          });
    });
  }

  m_work_queue.submit([&](handler &cgh) {
    auto managed_allocd_count_device =
        m_managed_allocd_count.get_access<access::mode::discard_write>(cgh);
    auto mem_pool_acc =
        m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
    auto garbage_collector_handle_acc =
        m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);
    cgh.single_task<class get_managed_allocd_count>(
        [managed_allocd_count_device, mem_pool_acc,
         garbage_collector_handle_acc] {
          auto gc_ref =
              mem_pool_acc[0].deref_handle(garbage_collector_handle_acc[0]);
          managed_allocd_count_device[0] =
              gc_ref->get_managed_allocation_count();
        });
  });

  Index_t managed_allocd_size;
  {
    auto managed_allocd_count_host =
        m_managed_allocd_count.get_access<access::mode::read>();
    managed_allocd_size = managed_allocd_count_host[0];
  }
  if (managed_allocd_size > 0) {
    // Expand markings
    bool was_markings_expanded_in_last_pass = true;
    m_markings_expanded.get_access<access::mode::discard_write>()[0] = false;

    while (was_markings_expanded_in_last_pass) {
      m_work_queue.submit([&](handler &cgh) {
        auto garbage_collector_handle_acc =
            m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);
        auto mem_pool_acc =
            m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
        auto markings_expanded_acc =
            m_markings_expanded.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class gc_mark>(
            cl::sycl::range<1>(managed_allocd_size),
            [garbage_collector_handle_acc, mem_pool_acc,
             markings_expanded_acc](item<1> itm) {
              auto gc_ref =
                  mem_pool_acc[0].deref_handle(garbage_collector_handle_acc[0]);
              const auto were_markings_expanded_here = gc_ref->run_mark_pass(
                  static_cast<Index_t>(itm.get_linear_id()), mem_pool_acc);
              if (were_markings_expanded_here) {
                markings_expanded_acc[0] = true;
              }
            });
      });
      {
        auto markings_expanded_host_acc =
            m_markings_expanded.get_access<access::mode::read_write>();
        was_markings_expanded_in_last_pass = markings_expanded_host_acc[0];
        markings_expanded_host_acc[0] = false;
      }
    }
    // Sweep unmarked, one pass
    m_work_queue.submit([&](handler &cgh) {
      auto garbage_collector_handle_acc =
          m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);
      auto mem_pool_acc =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class gc_sweep>(
          cl::sycl::range<1>(managed_allocd_size),
          [garbage_collector_handle_acc, mem_pool_acc](item<1> itm) {
            auto gc_ref =
                mem_pool_acc[0].deref_handle(garbage_collector_handle_acc[0]);
            gc_ref->sweep(static_cast<Index_t>(itm.get_linear_id()),
                          mem_pool_acc);
          });
    });

    m_work_queue.submit([&](handler &cgh) {
      auto garbage_collector_handle_acc =
          m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);
      auto mem_pool_acc =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class gc_prepare_compact>(
          [garbage_collector_handle_acc, mem_pool_acc]() {
            auto gc_ref =
                mem_pool_acc[0].deref_handle(garbage_collector_handle_acc[0]);
            gc_ref->reset_allocation_count();
          });
    });

    // Compact
    m_work_queue.submit([&](handler &cgh) {
      auto garbage_collector_handle_acc =
          m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);
      auto mem_pool_acc =
          m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class gc_compact>(
          cl::sycl::range<1>(managed_allocd_size),
          [garbage_collector_handle_acc, mem_pool_acc](item<1> itm) {
            auto gc_ref =
                mem_pool_acc[0].deref_handle(garbage_collector_handle_acc[0]);
            gc_ref->compact(static_cast<Index_t>(itm.get_linear_id()));
          });
    });
  }
}

CPUEvaluator::RuntimeBlock_t::RuntimeValue
CPUEvaluator::evaluate_program(const Compiler::ASTNodeHandle &root_node,
                               Index_t &max_concurrent_blocks_during_exec) {
  const auto begin_time = std::chrono::high_resolution_clock::now();
  create_first_block(root_node);

  max_concurrent_blocks_during_exec = 0;
  m_block_error_idx.get_access<access::mode::discard_write>()[0] = 0;
  m_requires_garbage_collection.get_access<access::mode::discard_write>()[0] =
      false;
  try {
    Index_t num_steps = 0;
    while (true) {
      ++num_steps;
      m_work_queue.submit([&](handler &cgh) {
        auto num_active_blocks_write =
            m_num_active_blocks_buff.get_access<access::mode::discard_write>(
                cgh);
        auto dep_tracker_acc =
            m_dependency_tracker_buff.get_access<access::mode::read_write>(cgh);
        cgh.single_task<class update_active_block_count>(
            [num_active_blocks_write, dep_tracker_acc]() {
              num_active_blocks_write[0] =
                  dep_tracker_acc[0].get_active_block_count();
              dep_tracker_acc[0].flip_active_blocks_buffer();
            });
      });

      check_for_block_errors(max_concurrent_blocks_during_exec);

      const auto num_active_blocks =
          m_num_active_blocks_buff.get_access<access::mode::read>()[0];
      max_concurrent_blocks_during_exec =
          std::max(num_active_blocks, max_concurrent_blocks_during_exec);
      {
        auto requires_garbage_collection_acc =
            m_requires_garbage_collection
                .get_access<access::mode::read_write>();
        if (requires_garbage_collection_acc[0] || num_active_blocks == 0) {
          perform_garbage_collection(num_active_blocks);
          requires_garbage_collection_acc[0] = false;
        }
      }

      if (num_active_blocks == 0) {
        break;
      }
      // main eval pass
      m_work_queue.submit([&](handler &cgh) {
        auto mem_pool_write =
            m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
        auto dependency_tracker =
            m_dependency_tracker_buff.get_access<access::mode::read_write>(cgh);
        auto garbage_collector_handle_acc =
            m_garbage_collector_handle_buff.get_access<access::mode::read>(cgh);

        auto block_error_idx_atomic_acc =
            m_block_error_idx.get_access<access::mode::read_write>(cgh);
        auto block_error_acc =
            m_errors_per_block.get_access<access::mode::read_write>(cgh);
        auto requires_garbage_collection_acc =
            m_requires_garbage_collection
                .get_access<access::mode::discard_write>(cgh);
        cgh.parallel_for<class run_eval_pass>(
            cl::sycl::range<1>(num_active_blocks),
            [dependency_tracker, mem_pool_write, block_error_idx_atomic_acc,
             block_error_acc, garbage_collector_handle_acc,
             requires_garbage_collection_acc](item<1> itm) {
              auto current_block =
                  dependency_tracker[0].get_block_at_index(itm.get_linear_id());
              auto derefd_current_block =
                  mem_pool_write[0].deref_handle(current_block);
              auto gc_ref = mem_pool_write[0].deref_handle(
                  garbage_collector_handle_acc[0]);
              derefd_current_block->set_resources(mem_pool_write,
                                                  dependency_tracker);
              const auto error = derefd_current_block->PerformEvalPass();
              switch (error.GetType()) {
              case Error::Type::Success:
                return;
              case Error::Type::GCOutOfSlots:
              case Error::Type::MemPoolAllocFailure:
                requires_garbage_collection_acc[0] = true;
                if (const auto re_append_error =
                        dependency_tracker[0].add_active_block(current_block);
                    re_append_error.GetType() == Error::Type::Success) {
                  return;
                }
                break;
              case Error::Type::InvalidArgType:
              case Error::Type::InvalidASTType:
              case Error::Type::ArityMismatch:
              case Error::Type::InvalidIndex:
              case Error::Type::EvaluatorOutOfActiveBlocks:
              case Error::Type::EvaluatorOutOfDeletionBlocks:
                break;
              }
              cl::sycl::atomic_ref<
                  Index_t, cl::sycl::memory_order::seq_cst,
                  cl::sycl::memory_scope::device,
                  cl::sycl::access::address_space::global_space>
                  block_error_idx_atomic(block_error_idx_atomic_acc[0]);
              block_error_acc[block_error_idx_atomic.fetch_add(1U)] = error;
            });
      });
    }
    std::cout << "Num steps: " << num_steps << std::endl;
  } catch (cl::sycl::exception e) {
    std::cerr << "Sycl exception: " << e.what() << std::endl;
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  std::cout << "total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     begin_time)
                   .count()
            << std::endl;
  m_work_queue.submit([&](handler &cgh) {
    auto mem_pool_acc =
        m_mem_pool_buff.get_access<access::mode::read_write>(cgh);
    auto result_on_host_acc =
        m_result_buffer_on_host.get_access<access::mode::discard_write>(cgh);
    auto result_val_ref_acc =
        m_result_value_buff.get_access<access::mode::read>(cgh);
    cgh.single_task<class fetch_result>(
        [result_on_host_acc, mem_pool_acc, result_val_ref_acc]() {
          result_on_host_acc[0] =
              *mem_pool_acc[0].deref_handle(result_val_ref_acc[0]);
        });
  });
  {
    auto result_on_host_acc =
        m_result_buffer_on_host.get_access<access::mode::read>();
    return result_on_host_acc[0];
  }
}
} // namespace FunGPU
