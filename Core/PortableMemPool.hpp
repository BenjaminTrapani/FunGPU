#pragma once

#include <CL/sycl.hpp>
#include "Types.h"
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace FunGPU {
/**
 * Arena-based memory pool that can be moved around without invalidating
 * references to objects in it.
 * Required to alloc compiled nodes on host and reference them on device.
 */

class PortableMemPool {
public:
  using DeviceAccessor_t =
      cl::sycl::accessor<PortableMemPool, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;
  using HostAccessor_t =
      cl::sycl::accessor<PortableMemPool, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::host_buffer>;
  template <class T> class Handle {
  public:
    Handle()
        : m_allocIndex(std::numeric_limits<Index_t>::max()),
          m_allocSize(std::numeric_limits<Index_t>::max()) {}
    Handle(const Index_t allocIndex, const Index_t allocSize)
        : m_allocIndex(allocIndex), m_allocSize(allocSize) {}

    template <class OtherT> Handle(const Handle<OtherT> &other) {
      *this = other;
    }

    Index_t GetAllocIndex() const { return m_allocIndex; }

    Index_t GetAllocSize() const { return m_allocSize; }
    bool operator==(const Handle &other) const {
      return m_allocIndex == other.m_allocIndex &&
             m_allocSize == other.m_allocSize;
    }
    bool operator!=(const Handle &other) const { return !(*this == other); }

    template <class OtherT> void operator=(const Handle<OtherT> &other) {
      static_assert(std::is_base_of<T, OtherT>::value ||
                        std::is_base_of<OtherT, T>::value,
                    "Cannot assign handle to handle of unrelated type");
      m_allocIndex = other.GetAllocIndex();
      m_allocSize = other.GetAllocSize();
    }

  private:
    Index_t m_allocIndex;
    Index_t m_allocSize;
  };

  template <class T> class ArrayHandle {
    friend class PortableMemPool;

  public:
    ArrayHandle() : m_count(std::numeric_limits<Index_t>::max()) {}
    ArrayHandle(const Index_t allocIndex, const Index_t allocSize,
                const Index_t count)
        : m_handle(allocIndex, allocSize), m_count(count) {}

    Index_t GetCount() const { return m_count; }

  private:
    Handle<T> m_handle;
    Index_t m_count;
  };

  // Implementers must define a public m_handle member.
  // Can't define it for you here because then your class would not be a
  // standard layout class :(
  template <class T> class EnableHandleFromThis {
    // Handle<T> m_handle;
  };

  template <class T, class... Args_t> Handle<T> Alloc(const Args_t &... args) {
    const auto handle =
        AllocImpl<T>(m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
    if (handle == Handle<T>()) {
      return handle;
    }

    auto* derefdAllocd = derefHandle(handle);
    // invoke T's constructor via placement new on allocated bytes
    auto allocdT = new (derefdAllocd) T(args...);

    using SetHandleFunctor_t = typename std::conditional<
        std::is_base_of<EnableHandleFromThis<T>, T>::value, SetHandleReal<T>,
        SetHandleNoOp<T>>::type;
    SetHandleFunctor_t::SetHandle(*derefdAllocd, handle);

    return handle;
  }

  template <class T>
  ArrayHandle<T> AllocArray(const Index_t arraySize,
                            const T &initialValue = T()) {
    return AllocArrayImpl<T>(arraySize, initialValue, m_smallBin, m_mediumBin,
                             m_largeBin, m_extraLargeBin);
  }

  template <class T> void Dealloc(const Handle<T> &handle) {
    DeallocImpl(handle, m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
  }

  template <class T> void DeallocArray(const ArrayHandle<T> &arrayHandle) {
    DeallocArrayImpl(arrayHandle, m_smallBin, m_mediumBin, m_largeBin,
                     m_extraLargeBin);
  }

  template <class T> T *derefHandle(const Handle<T> &handle) {
    return derefHandleImpl(handle, m_smallBin, m_mediumBin, m_largeBin,
                           m_extraLargeBin);
  }

  template <class T> T *derefHandle(const ArrayHandle<T> &handle) {
    return derefHandle(handle.m_handle);
  }

  Index_t GetTotalAllocationCount() {
    return GetTotalAllocationCountImpl(m_smallBin, m_mediumBin, m_largeBin,
                                       m_extraLargeBin);
  }

  template <class T> Index_t GetNumFree() {
    return GetNumFreeImpl<T>(m_smallBin, m_mediumBin, m_largeBin,
                             m_extraLargeBin);
  }

private:
  template <Index_t AllocSize_i, Index_t TotalBytes_i> struct Arena {
    static constexpr Index_t AllocSize = AllocSize_i;
    static constexpr Index_t TotalBytes = TotalBytes_i;

    static_assert(
        TotalBytes_i % AllocSize_i == 0,
        "Expected TotalBytes to be a multiple of the allocation size");

    static constexpr Index_t TotalNodes = TotalBytes_i / AllocSize_i;
    static_assert(
        (static_cast<size_t>(std::numeric_limits<Index_t>::max()) + 1) %
                TotalNodes ==
            0,
        "Ring buffer counts must wrap to 0 at multiples of total nodes");

    Arena() {
      for (Index_t i = 0; i < m_allocBeginIndices.size(); ++i) {
        m_allocBeginIndices[i] = i * AllocSize_i;
      }
    }

    Index_t AllocFromArena() {
      cl::sycl::atomic<Index_t> allocCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_totalAllocations)));
      const auto prevAllocCount = allocCount.fetch_add(1);
      if (prevAllocCount >= m_allocBeginIndices.size()) {
        return std::numeric_limits<Index_t>::max();
      }
      cl::sycl::atomic<Index_t> freeBlockBegin(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_freeBlockBegin)));
      const auto allocdIndex =
          freeBlockBegin.fetch_add(1) % m_allocBeginIndices.size();
      return m_allocBeginIndices[allocdIndex];
    }

    void FreeFromArena(const Index_t byteIdx) {
      cl::sycl::atomic<Index_t> allocdBlockBegin(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_allocdBlockBegin)));
      const auto freeDst =
          allocdBlockBegin.fetch_add(1) % m_allocBeginIndices.size();
      m_allocBeginIndices[freeDst] = byteIdx;

      cl::sycl::atomic<Index_t> allocCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_totalAllocations)));
      allocCount.fetch_sub(1);
    }

    unsigned char *GetBytes(const Index_t byteIdx) { return &m_bytes[byteIdx]; }

    std::array<unsigned char, TotalBytes_i> m_bytes;
    std::array<Index_t, TotalBytes_i / AllocSize_i> m_allocBeginIndices;

    Index_t m_freeBlockBegin = 0;
    Index_t m_allocdBlockBegin = 0;

    Index_t m_totalAllocations = 0;
  };

  template <class T> struct SetHandleNoOp {
    static void SetHandle(T &, const Handle<T> &) {}
  };

  template <class T> struct SetHandleReal {
    static void SetHandle(EnableHandleFromThis<T> &target,
                          const Handle<T> &handle) {
      auto targetAsSubclass = static_cast<T *>(&target);
      targetAsSubclass->m_handle = handle;
    }
  };

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  Handle<T> AllocImpl(Arena<allocSize, totalSize> &arena,
                      Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize >= sizeof(T)) {
      const auto allocdIdx = arena.AllocFromArena();
      if (allocdIdx == std::numeric_limits<Index_t>::max()) {
        return Handle<T>();
      }

      const Handle<T> resultHandle(
          allocdIdx, std::remove_reference<decltype(arena)>::type::AllocSize);
      return resultHandle;
    } else {
      return AllocImpl<T>(arenas...);
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  Handle<T> AllocImpl(Arena<allocSize, totalSize> &arena) {
    if (allocSize >= sizeof(T)) {
      const auto allocdIdx = arena.AllocFromArena();
      if (allocdIdx == std::numeric_limits<Index_t>::max()) {
        return Handle<T>();
      }
      const Handle<T> resultHandle(
          allocdIdx, std::remove_reference<decltype(arena)>::type::AllocSize);
      return resultHandle;
    } else {
      return Handle<T>();
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  ArrayHandle<T> AllocArrayImpl(const Index_t arraySize, const T &initialValue,
                                Arena<allocSize, totalSize> &arena,
                                Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize >= sizeof(T) * arraySize) {
      const auto allocdIdx = arena.AllocFromArena();
      if (allocdIdx == std::numeric_limits<Index_t>::max()) {
        return ArrayHandle<T>();
      }
      const ArrayHandle<T> resultHandle(
          allocdIdx, std::remove_reference<decltype(arena)>::type::AllocSize,
          arraySize);

      auto tAlignedValues = derefHandle(resultHandle);
      for (Index_t i = 0; i < arraySize; ++i) {
        tAlignedValues[i] = *(new (tAlignedValues + i) T(initialValue));
      }

      return resultHandle;
    } else {
      return AllocArrayImpl(arraySize, initialValue, arenas...);
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  ArrayHandle<T> AllocArrayImpl(const Index_t arraySize, const T &initialValue,
                                Arena<allocSize, totalSize> &arena) {
    if (allocSize >= sizeof(T) * arraySize) {
      const auto allocdIdx = arena.AllocFromArena();
      if (allocdIdx == std::numeric_limits<Index_t>::max()) {
        return ArrayHandle<T>();
      }
      const ArrayHandle<T> resultHandle(
          allocdIdx, std::remove_reference<decltype(arena)>::type::AllocSize,
          arraySize);

      auto tAlignedValues = derefHandle(resultHandle);
      for (Index_t i = 0; i < arraySize; ++i) {
        tAlignedValues[i] = *(new (tAlignedValues + i) T(initialValue));
      }

      return resultHandle;
    } else {
      return ArrayHandle<T>();
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  void DeallocImpl(const Handle<T> &handle, Arena<allocSize, totalSize> &arena,
                   Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize == handle.GetAllocSize()) {
      auto derefedForHandle = derefHandle(handle);
      // Explicitly call destructor
      derefedForHandle->~T();
      arena.FreeFromArena(handle.GetAllocIndex());
    } else {
      DeallocImpl(handle, arenas...);
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  void DeallocImpl(const Handle<T> &handle,
                   Arena<allocSize, totalSize> &arena) {
    if (allocSize == handle.GetAllocSize()) {
      auto derefedForHandle = derefHandle(handle);
      // Explicitly call destructor
      derefedForHandle->~T();
      arena.FreeFromArena(handle.GetAllocIndex());
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  void DeallocArrayImpl(const ArrayHandle<T> &arrayHandle,
                        Arena<allocSize, totalSize> &arena,
                        Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize == arrayHandle.m_handle.GetAllocSize()) {
      const auto &handle = arrayHandle.m_handle;
      auto derefdHandle = derefHandle(handle);
      for (Index_t i = 0; i < arrayHandle.GetCount(); ++i) {
        (derefdHandle[i]).~T();
      }

      arena.FreeFromArena(handle.GetAllocIndex());
    } else {
      DeallocArrayImpl(arrayHandle, arenas...);
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  void DeallocArrayImpl(const ArrayHandle<T> &arrayHandle,
                        Arena<allocSize, totalSize> &arena) {
    if (allocSize == arrayHandle.m_handle.GetAllocSize()) {
      const auto &handle = arrayHandle.m_handle;
      auto derefdHandle = derefHandle(handle);
      for (Index_t i = 0; i < arrayHandle.GetCount(); ++i) {
        (derefdHandle[i]).~T();
      }

      arena.FreeFromArena(handle.GetAllocIndex());
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  T *derefHandleImpl(const Handle<T> &handle,
                     Arena<allocSize, totalSize> &arena,
                     Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize == handle.GetAllocSize()) {
      auto bytesForIndex = arena.GetBytes(handle.GetAllocIndex());
      return reinterpret_cast<T *>(bytesForIndex);
    }

    return derefHandleImpl(handle, arenas...);
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  T *derefHandleImpl(const Handle<T> &handle,
                     Arena<allocSize, totalSize> &arena) {
    if (allocSize == handle.GetAllocSize()) {
      auto bytesForIndex = arena.GetBytes(handle.GetAllocIndex());
      return reinterpret_cast<T *>(bytesForIndex);
    } else {
      return nullptr;
    }
  }

  template <Index_t allocSize, Index_t totalSize, Index_t... allocSizes,
            Index_t... totalSizes>
  Index_t
  GetTotalAllocationCountImpl(Arena<allocSize, totalSize> &arena,
                              Arena<allocSizes, totalSizes> &... arenas) {
    Index_t totalAllocCount = 0;

    cl::sycl::atomic<Index_t> totalAllocations(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &arena.m_totalAllocations)));
    totalAllocCount += totalAllocations.load();
    totalAllocCount += GetTotalAllocationCountImpl(arenas...);

    return totalAllocCount;
  }

  template <Index_t allocSize, Index_t totalSize>
  Index_t GetTotalAllocationCountImpl(Arena<allocSize, totalSize> &arena) {
    cl::sycl::atomic<Index_t> totalAllocations(
        (cl::sycl::multi_ptr<Index_t,
                             cl::sycl::access::address_space::global_space>(
            &arena.m_totalAllocations)));
    return totalAllocations.load();
  }

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  Index_t GetNumFreeImpl(Arena<allocSize, totalSize> &arena,
                         Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize >= sizeof(T)) {
      cl::sycl::atomic<Index_t> totalAllocations(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &arena.m_totalAllocations)));
      return Arena<allocSize, totalSize>::TotalNodes - totalAllocations.load();
    } else {
      return GetNumFreeImpl<T>(arenas...);
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  Index_t GetNumFreeImpl(Arena<allocSize, totalSize> &arena) {
    if (allocSize >= sizeof(T)) {
      cl::sycl::atomic<Index_t> totalAllocations(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &arena.m_totalAllocations)));
      return Arena<allocSize, totalSize>::TotalNodes - totalAllocations.load();
    }
    return 0;
  }

  static constexpr Index_t binSize = 16777216;

  Arena<sizeof(int), binSize> m_smallBin;
  Arena<sizeof(int) * 8, binSize> m_mediumBin;
  Arena<sizeof(int) * 128, binSize> m_largeBin;
  Arena<sizeof(int) * 2097152, binSize> m_extraLargeBin;
};
} // namespace FunGPU
