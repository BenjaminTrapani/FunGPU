//
// Created by benjamintrapani on 8/10/19.
//

#pragma once

#include "PortableMemPool.hpp"
#include "SYCL/sycl.hpp"

namespace FunGPU {
template <class T, Index_t maxManagedAllocationsCount> class GarbageCollector {
public:
  GarbageCollector(const PortableMemPool::DeviceAccessor_t memPoolAcc)
      : m_memPoolAcc(memPoolAcc), m_managedAllocationsCountData(0),
        m_managedHandlesIdx(0) {}

  template <class... Args_t>
  PortableMemPool::Handle<T> AllocManaged(const Args_t &... args) {
    const auto allocdHandle = m_memPoolAcc[0].template Alloc<T>(args...);
    cl::sycl::atomic<Index_t> allocCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    const auto indexToAlloc = allocCount.fetch_add(1);
    if (indexToAlloc >= maxManagedAllocationsCount) {
      return PortableMemPool::Handle<T>();
    }
    m_managedHandles[m_managedHandlesIdx][indexToAlloc] = allocdHandle;
    return allocdHandle;
  }

  Index_t GetManagedAllocationCount() {
    cl::sycl::atomic<Index_t> allocCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    return allocCount.load();
  }

  void SetMemPoolAcc(const PortableMemPool::DeviceAccessor_t &memPoolAcc) {
    m_memPoolAcc = memPoolAcc;
  }

  bool RunMarkPass(const Index_t idx) {
    const auto &handleForIdx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefdForHandle = m_memPoolAcc[0].derefHandle(handleForIdx);
    return derefdForHandle->ExpandMarkings();
  }

  void Sweep(const Index_t idx) {
    auto &handleForIdx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefdForHandle = m_memPoolAcc[0].derefHandle(handleForIdx);
    if (!derefdForHandle->GetIsMarked()) {
      m_memPoolAcc[0].Dealloc(handleForIdx);
      handleForIdx = PortableMemPool::Handle<T>();
    } else {
      derefdForHandle->ClearMarking();
    }
  }

  void ResetAllocationCount() {
    cl::sycl::atomic<Index_t> allocCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    allocCount.store(0);
    m_prevManagedHandlesIdx = m_managedHandlesIdx;
    m_managedHandlesIdx = (m_managedHandlesIdx + 1) % m_managedHandles.size();
  }

  void Compact(const Index_t idx) {
    const auto &handleForIdx = m_managedHandles[m_prevManagedHandlesIdx][idx];
    if (handleForIdx != PortableMemPool::Handle<T>()) {
      cl::sycl::atomic<Index_t> allocCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_managedAllocationsCountData)));
      m_managedHandles[m_managedHandlesIdx][allocCount.fetch_add(1)] =
          handleForIdx;
    }
  }

private:
  PortableMemPool::DeviceAccessor_t m_memPoolAcc;
  Index_t m_managedAllocationsCountData;
  std::array<std::array<PortableMemPool::Handle<T>, maxManagedAllocationsCount>,
             2>
      m_managedHandles;
  Index_t m_managedHandlesIdx;
  Index_t m_prevManagedHandlesIdx;
};
}
