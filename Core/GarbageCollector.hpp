//
// Created by benjamintrapani on 8/10/19.
//

#pragma once

#include "Error.hpp"
#include "PortableMemPool.hpp"
#include <CL/sycl.hpp>

namespace FunGPU {
template <class T, Index_t maxManagedAllocationsCount> class GarbageCollector {
public:
  GarbageCollector()
      : m_managedAllocationsCountData(0),
        m_managedHandlesIdx(0) {}

  template <class... Args_t>
  Error AllocManaged(const PortableMemPool::DeviceAccessor_t& memPoolAcc, PortableMemPool::Handle<T> &result, Args_t &&... args) {
    cl::sycl::atomic<Index_t> allocCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    result = memPoolAcc[0].template Alloc<T>(std::forward<Args_t>(args)...);
    if (result == PortableMemPool::Handle<T>()) {
      return Error(Error::Type::MemPoolAllocFailure);
    }
    auto derefdResult = memPoolAcc[0].derefHandle(result);
    if (const auto error = derefdResult->Init(); error.GetType() != Error::Type::Success) {
      memPoolAcc[0].Dealloc(result);
      return error;
    }
    const auto indexToAlloc = allocCount.fetch_add(1);
    if (indexToAlloc >= maxManagedAllocationsCount) {
      memPoolAcc[0].Dealloc(result);
      return Error(Error::Type::GCOutOfSlots);
    }

    m_managedHandles[m_managedHandlesIdx][indexToAlloc] = result;
    return Error();
  }

  Index_t GetManagedAllocationCount() {
    cl::sycl::atomic<Index_t> allocCount(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &m_managedAllocationsCountData)));
    return std::min(allocCount.load(), maxManagedAllocationsCount);
  }

  bool RunMarkPass(const Index_t idx, const PortableMemPool::DeviceAccessor_t& memPoolAcc) {
    const auto &handleForIdx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefdForHandle = memPoolAcc[0].derefHandle(handleForIdx);
    return derefdForHandle->ExpandMarkings();
  }

  void Sweep(const Index_t idx, const PortableMemPool::DeviceAccessor_t& memPoolAcc) {
    auto &handleForIdx = m_managedHandles[m_managedHandlesIdx][idx];
    auto derefdForHandle = memPoolAcc[0].derefHandle(handleForIdx);
    if (!derefdForHandle->GetIsMarked()) {
      memPoolAcc[0].Dealloc(handleForIdx);
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
  Index_t m_managedAllocationsCountData;
  std::array<std::array<PortableMemPool::Handle<T>, maxManagedAllocationsCount>,
             2>
      m_managedHandles;
  Index_t m_managedHandlesIdx;
  Index_t m_prevManagedHandlesIdx;
};
} // namespace FunGPU
