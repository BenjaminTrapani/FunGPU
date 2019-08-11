#pragma once

#include "SYCL/sycl.hpp"
#include "SpinLock.h"
#include <atomic>
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
        : m_allocIndex(std::numeric_limits<size_t>::max()),
          m_allocSize(std::numeric_limits<size_t>::max()) {}
    Handle(const size_t allocIndex, const size_t allocSize)
        : m_allocIndex(allocIndex), m_allocSize(allocSize) {}

    template <class OtherT> Handle(const Handle<OtherT> &other) {
      *this = other;
    }

    size_t GetAllocIndex() const { return m_allocIndex; }

    size_t GetAllocSize() const { return m_allocSize; }
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
    size_t m_allocIndex;
    size_t m_allocSize;
  };

  template <class T> class ArrayHandle {
    friend class PortableMemPool;

  public:
    ArrayHandle() : m_count(std::numeric_limits<size_t>::max()) {}
    ArrayHandle(const size_t allocIndex, const size_t allocSize,
                const size_t count)
        : m_handle(allocIndex, allocSize), m_count(count) {}

    size_t GetCount() const { return m_count; }

  private:
    Handle<T> m_handle;
    size_t m_count;
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

    auto bytesForHandle =
        reinterpret_cast<unsigned char *>(derefHandle(handle));
    // invoke T's constructor via placement new on allocated bytes
    auto allocdT = new (bytesForHandle) T(args...);

    using SetHandleFunctor_t = typename std::conditional<
        std::is_base_of<EnableHandleFromThis<T>, T>::value, SetHandleReal<T>,
        SetHandleNoOp<T>>::type;
    auto derefedAllocd = derefHandle(handle);
    SetHandleFunctor_t::SetHandle(*derefedAllocd, handle);

    return handle;
  }

  template <class T>
  ArrayHandle<T> AllocArray(const size_t arraySize,
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

  size_t GetTotalAllocationCount() {
    return GetTotalAllocationCountImpl(m_smallBin, m_mediumBin, m_largeBin,
                                       m_extraLargeBin);
  }

private:
  struct ListNode {
    ListNode(const size_t beginIndex) : m_beginIndex(beginIndex) {}
    ListNode() : m_beginIndex(0), m_nextNode(0) {}
    bool operator==(const ListNode &other) const {
      return m_beginIndex == other.m_beginIndex &&
             m_nextNode == other.m_nextNode &&
             m_indexInStorage == other.m_indexInStorage;
    }

    size_t m_beginIndex;
    size_t m_nextNode;
    size_t m_indexInStorage;
  };

  template <size_t AllocSize_i, size_t TotalBytes_i> struct Arena {
    static constexpr size_t AllocSize = AllocSize_i;
    static constexpr size_t TotalBytes = TotalBytes_i;

    static_assert(
        TotalBytes_i % AllocSize_i == 0,
        "Expected TotalBytes to be a multiple of the allocation size");

    static constexpr size_t TotalNodes = TotalBytes_i / AllocSize_i;

    Arena() {
      for (size_t i = 0; i < m_listNodeStorage.size(); ++i) {
        auto &listNode = m_listNodeStorage[i];
        listNode.m_beginIndex = i * AllocSize_i;
        listNode.m_indexInStorage = i;
        listNode.m_nextNode = i + 1;
      }

      m_freeListHead = m_listNodeStorage[0];
      m_freeListTail.m_beginIndex = m_bytes.size() + 1;
      m_freeListTail.m_nextNode = m_listNodeStorage.size() + 1;
    }

    ListNode AllocFromArena() {
      m_lock.Aquire();

      if (m_freeListHead == m_freeListTail) {
        m_lock.Release();
        return m_freeListHead;
      }

      const auto result = m_freeListHead;
      const auto nextListIndex = m_freeListHead.m_nextNode;
      if (nextListIndex >= m_listNodeStorage.size()) {
        m_freeListHead = m_freeListTail;
      } else {
        m_freeListHead = m_listNodeStorage[nextListIndex];
      }

      ++m_totalAllocations;

      m_lock.Release();

      return result;
    }

    void FreeFromArena(const size_t allocIndex) {
      m_lock.Aquire();

      m_listNodeStorage[allocIndex].m_nextNode =
          m_freeListHead.m_indexInStorage;
      m_freeListHead = m_listNodeStorage[allocIndex];

      --m_totalAllocations;

      m_lock.Release();
    }

    unsigned char *GetBytes(const size_t allocIndex) {
      const auto &listNodeForAllocIndex = m_listNodeStorage[allocIndex];
      return &m_bytes[listNodeForAllocIndex.m_beginIndex];
    }

    std::array<unsigned char, TotalBytes_i> m_bytes;
    std::array<ListNode, TotalBytes_i / AllocSize_i> m_listNodeStorage;

    ListNode m_freeListHead;
    ListNode m_freeListTail;
    SpinLock m_lock;

    size_t m_totalAllocations = 0;
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

  template <class T, size_t allocSize, size_t totalSize, size_t... allocSizes,
            size_t... totalSizes>
  Handle<T> AllocImpl(Arena<allocSize, totalSize> &arena,
                      Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize >= sizeof(T)) {
      const auto allocdListNode = arena.AllocFromArena();
      if (allocdListNode == arena.m_freeListTail) {
        return Handle<T>();
      }

      const Handle<T> resultHandle(
          allocdListNode.m_indexInStorage,
          std::remove_reference<decltype(arena)>::type::AllocSize);
      return resultHandle;
    } else {
      return AllocImpl<T>(arenas...);
    }
  }

  template <class T, size_t allocSize, size_t totalSize>
  Handle<T> AllocImpl(Arena<allocSize, totalSize> &arena) {
    if (allocSize >= sizeof(T)) {
      const auto allocdListNode = arena.AllocFromArena();
      if (allocdListNode == arena.m_freeListTail) {
        return Handle<T>();
      }
      const Handle<T> resultHandle(
          allocdListNode.m_indexInStorage,
          std::remove_reference<decltype(arena)>::type::AllocSize);
      return resultHandle;
    } else {
      return Handle<T>();
    }
  }

  template <class T, size_t allocSize, size_t totalSize, size_t... allocSizes,
            size_t... totalSizes>
  ArrayHandle<T> AllocArrayImpl(const size_t arraySize, const T &initialValue,
                                Arena<allocSize, totalSize> &arena,
                                Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize >= sizeof(T) * arraySize) {
      const auto allocdListNode = arena.AllocFromArena();
      if (allocdListNode == arena.m_freeListTail) {
        return ArrayHandle<T>();
      }
      const ArrayHandle<T> resultHandle(
          allocdListNode.m_indexInStorage,
          std::remove_reference<decltype(arena)>::type::AllocSize, arraySize);

      auto tAlignedValues = derefHandle(resultHandle);
      for (size_t i = 0; i < arraySize; ++i) {
        tAlignedValues[i] = *(new (tAlignedValues + i) T(initialValue));
      }

      return resultHandle;
    } else {
      return AllocArrayImpl(arraySize, initialValue, arenas...);
    }
  }

  template <class T, size_t allocSize, size_t totalSize>
  ArrayHandle<T> AllocArrayImpl(const size_t arraySize, const T &initialValue,
                                Arena<allocSize, totalSize> &arena) {
    if (allocSize >= sizeof(T) * arraySize) {
      const auto allocdListNode = arena.AllocFromArena();
      if (allocdListNode == arena.m_freeListTail) {
        return ArrayHandle<T>();
      }
      const ArrayHandle<T> resultHandle(
          allocdListNode.m_indexInStorage,
          std::remove_reference<decltype(arena)>::type::AllocSize, arraySize);

      auto tAlignedValues = derefHandle(resultHandle);
      for (size_t i = 0; i < arraySize; ++i) {
        tAlignedValues[i] = *(new (tAlignedValues + i) T(initialValue));
      }

      return resultHandle;
    } else {
      return ArrayHandle<T>();
    }
  }

  template <class T, size_t allocSize, size_t totalSize, size_t... allocSizes,
            size_t... totalSizes>
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

  template <class T, size_t allocSize, size_t totalSize>
  void DeallocImpl(const Handle<T> &handle,
                   Arena<allocSize, totalSize> &arena) {
    if (allocSize == handle.GetAllocSize()) {
      auto derefedForHandle = derefHandle(handle);
      // Explicitly call destructor
      derefedForHandle->~T();
      arena.FreeFromArena(handle.GetAllocIndex());
    }
  }

  template <class T, size_t allocSize, size_t totalSize, size_t... allocSizes,
            size_t... totalSizes>
  void DeallocArrayImpl(const ArrayHandle<T> &arrayHandle,
                        Arena<allocSize, totalSize> &arena,
                        Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize == arrayHandle.m_handle.GetAllocSize()) {
      const auto &handle = arrayHandle.m_handle;
      auto derefdHandle = derefHandle(handle);
      for (size_t i = 0; i < arrayHandle.GetCount(); ++i) {
        (derefdHandle[i]).~T();
      }

      arena.FreeFromArena(handle.GetAllocIndex());
    } else {
      DeallocArrayImpl(arrayHandle, arenas...);
    }
  }

  template <class T, size_t allocSize, size_t totalSize>
  void DeallocArrayImpl(const ArrayHandle<T> &arrayHandle,
                        Arena<allocSize, totalSize> &arena) {
    if (allocSize == arrayHandle.m_handle.GetAllocSize()) {
      const auto &handle = arrayHandle.m_handle;
      auto derefdHandle = derefHandle(handle);
      for (size_t i = 0; i < arrayHandle.GetCount(); ++i) {
        (derefdHandle[i]).~T();
      }

      arena.FreeFromArena(handle.GetAllocIndex());
    }
  }

  template <class T, size_t allocSize, size_t totalSize, size_t... allocSizes,
            size_t... totalSizes>
  T *derefHandleImpl(const Handle<T> &handle,
                     Arena<allocSize, totalSize> &arena,
                     Arena<allocSizes, totalSizes> &... arenas) {
    if (allocSize == handle.GetAllocSize()) {
      auto bytesForIndex = arena.GetBytes(handle.GetAllocIndex());
      return reinterpret_cast<T *>(bytesForIndex);
    }

    return derefHandleImpl(handle, arenas...);
  }

  template <class T, size_t allocSize, size_t totalSize>
  T *derefHandleImpl(const Handle<T> &handle,
                     Arena<allocSize, totalSize> &arena) {
    if (allocSize == handle.GetAllocSize()) {
      auto bytesForIndex = arena.GetBytes(handle.GetAllocIndex());
      return reinterpret_cast<T *>(bytesForIndex);
    } else {
      return nullptr;
    }
  }

  template <size_t allocSize, size_t totalSize, size_t... allocSizes,
            size_t... totalSizes>
  size_t
  GetTotalAllocationCountImpl(Arena<allocSize, totalSize> &arena,
                              Arena<allocSizes, totalSizes> &... arenas) {
    size_t totalAllocCount = 0;

    {
      arena.m_lock.Aquire();
      totalAllocCount += arena.m_totalAllocations;
      arena.m_lock.Release();
    }
    totalAllocCount += GetTotalAllocationCountImpl(arenas...);

    return totalAllocCount;
  }

  template <size_t allocSize, size_t totalSize>
  size_t GetTotalAllocationCountImpl(Arena<allocSize, totalSize> &arena) {
    arena.m_lock.Aquire();
    const auto result = arena.m_totalAllocations;
    arena.m_lock.Release();

    return result;
  }

  static constexpr size_t binSize = 16777216;

  Arena<sizeof(int), binSize> m_smallBin;
  Arena<sizeof(int) * 8, binSize> m_mediumBin;
  Arena<sizeof(int) * 128, binSize> m_largeBin;
  Arena<sizeof(int) * 4194304, binSize> m_extraLargeBin;
};
}
