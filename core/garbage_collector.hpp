//
// Created by benjamintrapani on 8/10/19.
//

#pragma once

#include "core/error.hpp"
#include "core/portable_mem_pool.hpp"
#include <CL/sycl.hpp>

namespace FunGPU {
template <class T, Index_t maxManagedAllocationsCount> class GarbageCollector {
public:
  GarbageCollector()
      : m_managedAllocationsCountData(0), m_managedHandlesIdx(0) {}

  template <class... Args_t>
  Error alloc_managed(const PortableMemPool::DeviceAccessor_t &mem_pool_acc,
                      PortableMemPool::Handle<T> &result, Args_t &&...args) {
    cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                         cl::sycl::memory_scope::device,
                         cl::sycl::access::address_space::global_space>
        allocCount(m_managedAllocationsCountData);
    result = mem_pool_acc[0].template alloc<T>(std::forward<Args_t>(args)...);
    if (result == PortableMemPool::Handle<T>()) {
      return Error(Error::Type::MemPoolAllocFailure);
    }
    auto derefdResult = mem_pool_acc[0].deref_handle(result);
    if (const auto error = derefdResult->init();
        error.GetType() != Error::Type::Success) {
      mem_pool_acc[0].dealloc(result);
      return error;
    }
    const auto indexToAlloc = allocCount.fetch_add(1U);
    if (indexToAlloc >= maxManagedAllocationsCount) {
      mem_pool_acc[0].dealloc(result);
      return Error(Error::Type::GCOutOfSlots);
    }

    m_managedHandles[m_managedHandlesIdx][indexToAlloc] = result;
    return Error();
  }

  Index_t get_managed_allocation_count() {
    cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                         cl::sycl::memory_scope::device,
                         cl::sycl::access::address_space::global_space>
        allocCount(m_managedAllocationsCountData);
    return std::min(allocCount.load(), maxManagedAllocationsCount);
  }

  bool run_mark_pass(const Index_t idx,
                     const PortableMemPool::DeviceAccessor_t &mem_pool_acc) {
    const auto &handleForIdx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefdForHandle = mem_pool_acc[0].deref_handle(handleForIdx);
    return derefdForHandle->expand_markings();
  }

  void sweep(const Index_t idx,
             const PortableMemPool::DeviceAccessor_t &mem_pool_acc) {
    auto &handle_for_idx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefd_for_handle = mem_pool_acc[0].deref_handle(handle_for_idx);
    if (!derefd_for_handle->get_is_marked()) {
      mem_pool_acc[0].dealloc(handle_for_idx);
      handle_for_idx = PortableMemPool::Handle<T>();
    } else {
      derefd_for_handle->clear_marking();
    }
  }

  void reset_allocation_count() {
    m_managedAllocationsCountData = 0;
    m_prevManagedHandlesIdx = m_managedHandlesIdx;
    m_managedHandlesIdx = (m_managedHandlesIdx + 1) % m_managedHandles.size();
  }

  void compact(const Index_t idx) {
    const auto &handleForIdx = m_managedHandles[m_prevManagedHandlesIdx][idx];
    if (handleForIdx != PortableMemPool::Handle<T>()) {
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          allocCount(m_managedAllocationsCountData);
      m_managedHandles[m_managedHandlesIdx][allocCount.fetch_add(1U)] =
          handleForIdx;
    }
  }

private:
  Index_t m_managedAllocationsCountData;
  std::array<std::array<PortableMemPool::Handle<T>, maxManagedAllocationsCount>,
             2>
      m_managedHandles;
  Index_t m_managedHandlesIdx;
  Index_t m_prevManagedHandlesIdx;
};
} // namespace FunGPU
