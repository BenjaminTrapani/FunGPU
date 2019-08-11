//
// Created by benjamintrapani on 8/10/19.
//

#pragma once

#include "PortableMemPool.hpp"
#include "SYCL/sycl.hpp"

namespace FunGPU {
template <class T, size_t maxManagedAllocationsCount> class GarbageCollector {
public:
  GarbageCollector(const PortableMemPool::DeviceAccessor_t memPoolAcc)
      : m_memPoolAcc(memPoolAcc), m_managedAllocationsCountData(0),
        m_managedHandlesIdx(0) {}

  template <class... Args_t>
  PortableMemPool::Handle<T> AllocManaged(const Args_t &... args) {
    const auto allocdHandle = m_memPoolAcc[0].template Alloc<T>(args...);
    cl::sycl::atomic<unsigned int> allocCount(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    const auto indexToAlloc = allocCount.fetch_add(1);
    if (indexToAlloc >= maxManagedAllocationsCount) {
      return PortableMemPool::Handle<T>();
    }
    m_managedHandles[m_managedHandlesIdx][indexToAlloc] = allocdHandle;
    return allocdHandle;
  }

  unsigned int GetManagedAllocationCount() {
    cl::sycl::atomic<unsigned int> allocCount(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    return allocCount.load();
  }

  void SetMemPoolAcc(const PortableMemPool::DeviceAccessor_t &memPoolAcc) {
    m_memPoolAcc = memPoolAcc;
  }

  bool RunMarkPass(const unsigned int idx) {
    const auto &handleForIdx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefdForHandle = m_memPoolAcc[0].derefHandle(handleForIdx);
    return derefdForHandle->ExpandMarkings();
  }

  void Sweep(const unsigned int idx) {
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
    cl::sycl::atomic<unsigned int> allocCount(
        (cl::sycl::multi_ptr<unsigned int,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    allocCount.store(0);
    m_prevManagedHandlesIdx = m_managedHandlesIdx;
    m_managedHandlesIdx = (m_managedHandlesIdx + 1) % m_managedHandles.size();
  }

  void Compact(const unsigned int idx) {
    const auto &handleForIdx = m_managedHandles[m_prevManagedHandlesIdx][idx];
    if (handleForIdx != PortableMemPool::Handle<T>()) {
      cl::sycl::atomic<unsigned int> allocCount(
          (cl::sycl::multi_ptr<unsigned int,
                               cl::sycl::access::address_space::global_space>(
              &m_managedAllocationsCountData)));
      m_managedHandles[m_managedHandlesIdx][allocCount.fetch_add(1)] =
          handleForIdx;
    }
  }

private:
  PortableMemPool::DeviceAccessor_t m_memPoolAcc;
  unsigned int m_managedAllocationsCountData;
  std::array<std::array<PortableMemPool::Handle<T>, maxManagedAllocationsCount>,
             2>
      m_managedHandles;
  unsigned int m_managedHandlesIdx;
  unsigned int m_prevManagedHandlesIdx;
};
}
