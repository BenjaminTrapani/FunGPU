#pragma once

#include "Core/Types.hpp"
#include <CL/sycl.hpp>
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
    Handle() : m_distFromMemPoolBase(std::numeric_limits<Index_t>::max()) {}
    Handle(const Index_t distFromMemPoolBase)
        : m_distFromMemPoolBase(distFromMemPoolBase) {}

    template <class OtherT> Handle(const Handle<OtherT> &other) {
      *this = other;
    }

    Index_t GetDistFromMemPoolBase() const { return m_distFromMemPoolBase; }

    bool operator==(const Handle &other) const {
      return m_distFromMemPoolBase == other.m_distFromMemPoolBase;
    }
    bool operator!=(const Handle &other) const { return !(*this == other); }
    bool operator<(const Handle &other) const {
      return m_distFromMemPoolBase < other.m_distFromMemPoolBase;
    }

    template <class OtherT> void operator=(const Handle<OtherT> &other) {
      static_assert(std::is_base_of<T, OtherT>::value ||
                        std::is_base_of<OtherT, T>::value,
                    "Cannot assign handle to handle of unrelated type");
      m_distFromMemPoolBase = other.GetDistFromMemPoolBase();
    }

  private:
    Index_t m_distFromMemPoolBase;
  };

  template <class T> class TrivialHandle {
  public:
    TrivialHandle() = default;
    TrivialHandle(const Handle<T> other)
        : m_distFromMemPoolBase(other.GetDistFromMemPoolBase()) {}

    TrivialHandle &operator=(const Handle<T> other) {
      m_distFromMemPoolBase = other.GetDistFromMemPoolBase();
      return *this;
    }

    Handle<T> unpack() const { return Handle<T>(m_distFromMemPoolBase); }

  private:
    Index_t m_distFromMemPoolBase;
  };

  template <class T> class ArrayHandle {
    friend class PortableMemPool;

  public:
    ArrayHandle() : m_count(0) {}
    ArrayHandle(const Index_t distFromMemPoolBase, const Index_t count)
        : m_handle(distFromMemPoolBase), m_count(count) {}

    Index_t GetCount() const { return m_count; }

    // Do not free the handle returned from this function, that will break the
    // allocator.
    Handle<T> ElementHandle(const Index_t elemIdx) const {
      return Handle<T>(m_handle.GetDistFromMemPoolBase() + elemIdx * sizeof(T));
    }

    bool operator==(const ArrayHandle<T> &other) const {
      return m_handle == other.m_handle && m_count == other.m_count;
    }

    bool operator!=(const ArrayHandle<T> &other) const {
      return !(*this == other);
    }

  private:
    Handle<T> m_handle;
    Index_t m_count;
  };

  template <class T> class TrivialArrayHandle {
  public:
    TrivialArrayHandle() = default;
    TrivialArrayHandle(const ArrayHandle<T> &handle)
        : m_handle(handle.ElementHandle(0)), m_count(handle.GetCount()) {}

    TrivialArrayHandle &operator=(const ArrayHandle<T> &other) {
      m_handle = other.m_handle;
      m_count = other.m_count;
      return *this;
    }

    ArrayHandle<T> unpack() const {
      return ArrayHandle<T>(m_handle.unpack().GetDistFromMemPoolBase(),
                            m_count);
    }

  private:
    TrivialHandle<T> m_handle;
    Index_t m_count;
  };

  // Implementers must define a public m_handle member.
  // Can't define it for you here because then your class would not be a
  // standard layout class :(
  template <class T> class EnableHandleFromThis {
    // Handle<T> m_handle;
  };

  template <class T, class... Args_t> Handle<T> Alloc(const Args_t &...args) {
    const auto handle =
        AllocImpl<T>(m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
    if (handle == Handle<T>()) {
      return handle;
    }

    auto *derefdAllocd = derefHandle(handle);
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
    if (arraySize == 0) {
      return ArrayHandle<T>();
    }
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
    return reinterpret_cast<T *>(reinterpret_cast<std::byte *>(this) +
                                 handle.GetDistFromMemPoolBase());
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
    static_assert(
        TotalBytes_i % AllocSize_i == 0,
        "Expected TotalBytes to be a multiple of the allocation size");

    static_assert(
        (static_cast<size_t>(std::numeric_limits<Index_t>::max()) + 1) %
                (TotalBytes_i / AllocSize_i) ==
            0,
        "Ring buffer counts must wrap to 0 at multiples of total nodes");

    Arena() {
      for (Index_t i = 0; i < m_allocBeginIndices.size(); ++i) {
        m_allocBeginIndices[i] = i * AllocSize_i;
      }
    }

    std::byte *AllocFromArena() {
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          allocCount(m_totalAllocations);
      const auto prevAllocCount = allocCount.fetch_add(1U);
      if (prevAllocCount >= m_allocBeginIndices.size()) {
        return nullptr;
      }
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          freeBlockBegin(m_freeBlockBegin);
      const auto allocdIndex =
          freeBlockBegin.fetch_add(1U) % m_allocBeginIndices.size();
      return &m_storage[m_allocBeginIndices[allocdIndex]];
    }

    bool OwnsData(const std::byte *data) const {
      return data >= &m_storage[0] &&
             data < &m_storage[TotalBytes_i - AllocSize_i];
    }

    void FreeFromArena(std::byte *data) {
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          allocdBlockBegin(m_allocdBlockBegin);
      const auto freeDst =
          allocdBlockBegin.fetch_add(1U) % m_allocBeginIndices.size();
      m_allocBeginIndices[freeDst] = data - &m_storage[0];

      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          allocCount(m_totalAllocations);
      allocCount.fetch_sub(1U);
    }

    alignas(AllocSize_i) std::byte m_storage[TotalBytes_i];
    std::array<Index_t, TotalBytes_i / AllocSize_i> m_allocBeginIndices;
    Index_t m_freeBlockBegin = 0;
    Index_t m_allocdBlockBegin = 0;

    Index_t m_totalAllocations = 0;
    Index_t m_reservedAllocations = 0;
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

  template <class T, class Pred, class CB, Index_t allocSize, Index_t totalSize>
  decltype(auto) WithArena(Pred &&pred, CB &&cb,
                           Arena<allocSize, totalSize> &arena) {
    if (pred(arena)) {
      return cb(arena);
    }
    return cb();
  }

  template <class T, class Pred, class CB, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  decltype(auto) WithArena(Pred &&pred, CB &&cb,
                           Arena<allocSize, totalSize> &arena,
                           Arena<allocSizes, totalSizes> &...arenas) {
    if (pred(arena)) {
      return cb(arena);
    }
    return WithArena<T, Pred, CB>(std::forward<Pred>(pred),
                                  std::forward<CB>(cb), arenas...);
  }

  template <class T> struct AllocPred {
    template <Index_t allocSize, Index_t totalSize>
    constexpr bool operator()(const Arena<allocSize, totalSize> &) {
      return sizeof(T) <= allocSize;
    }
  };

  template <class T> struct AllocHandler {
    AllocHandler(PortableMemPool &memPool) : m_memPool(memPool) {}

    Handle<T> operator()() { return Handle<T>(); }

    template <class DispatchedArena>
    Handle<T> operator()(DispatchedArena &arena) {
      const auto *data = arena.AllocFromArena();
      if (data == nullptr) {
        return Handle<T>();
      }

      return Handle<T>(data - reinterpret_cast<std::byte *>(&m_memPool));
    }

    PortableMemPool &m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  Handle<T> AllocImpl(Arena<allocSizes, totalSizes> &...arenas) {
    return WithArena<T>(AllocPred<T>(), AllocHandler<T>(*this), arenas...);
  }

  template <class T> struct AllocArrayPred {
    AllocArrayPred(const Index_t arraySize) : m_arraySize(arraySize) {}

    template <Index_t allocSize, Index_t totalSize>
    bool operator()(const Arena<allocSize, totalSize> &arena) {
      return allocSize >= sizeof(T) * m_arraySize;
    }

    const Index_t m_arraySize;
  };

  template <class T> struct AllocArrayHandler {
    AllocArrayHandler(const Index_t arraySize, const T &initialValue,
                      PortableMemPool &memPool)
        : m_arraySize(arraySize), m_initialValue(initialValue),
          m_memPool(memPool) {}

    ArrayHandle<T> operator()() { return ArrayHandle<T>(); }

    template <class DispatchedArena>
    ArrayHandle<T> operator()(DispatchedArena &arena) {
      auto *data = arena.AllocFromArena();
      if (data == nullptr) {
        return ArrayHandle<T>();
      }

      auto tAlignedValues = reinterpret_cast<T *>(data);
      for (Index_t i = 0; i < m_arraySize; ++i) {
        tAlignedValues[i] = *(new (tAlignedValues + i) T(m_initialValue));
      }

      return ArrayHandle<T>(data - reinterpret_cast<std::byte *>(&m_memPool),
                            m_arraySize);
    }

    const Index_t m_arraySize;
    const T &m_initialValue;
    PortableMemPool &m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  ArrayHandle<T> AllocArrayImpl(const Index_t arraySize, const T &initialValue,
                                Arena<allocSizes, totalSizes> &...arenas) {
    return WithArena<T>(AllocArrayPred<T>(arraySize),
                        AllocArrayHandler<T>(arraySize, initialValue, *this),
                        arenas...);
  }

  template <class T> struct MatchArenaOffset {
    MatchArenaOffset(const PortableMemPool &memPool,
                     const Index_t distFromMemPoolBase)
        : m_distFromMemPoolBase(distFromMemPoolBase), m_memPool(memPool) {}

    template <Index_t allocSize, Index_t totalSize>
    bool operator()(const Arena<allocSize, totalSize> &arena) {
      return arena.OwnsData(reinterpret_cast<const std::byte *>(&m_memPool) +
                            m_distFromMemPoolBase);
    }

    const Index_t m_distFromMemPoolBase;
    const PortableMemPool &m_memPool;
  };

  template <class T> struct DeallocHandler {
    DeallocHandler(const Handle<T> &handle, PortableMemPool &memPool)
        : m_handle(handle), m_memPool(memPool) {}
    void operator()() {}

    template <class DispatchedArena> void operator()(DispatchedArena &arena) {
      auto derefdForHandle = m_memPool.derefHandle(m_handle);
      // Explicitly call destructor
      derefdForHandle->~T();
      arena.FreeFromArena(reinterpret_cast<std::byte *>(derefdForHandle));
    }

    Handle<T> m_handle;
    PortableMemPool &m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  void DeallocImpl(const Handle<T> &handle,
                   Arena<allocSizes, totalSizes> &...arenas) {
    WithArena<T>(MatchArenaOffset<T>(*this, handle.GetDistFromMemPoolBase()),
                 DeallocHandler<T>(handle, *this), arenas...);
  }

  template <class T> struct DeallocArrayHandler {
    DeallocArrayHandler(const ArrayHandle<T> &arrayHandle,
                        PortableMemPool &memPool)
        : m_arrayHandle(arrayHandle), m_memPool(memPool) {}

    void operator()() {}

    template <class DispatchedArena> void operator()(DispatchedArena &arena) {
      const auto &handle = m_arrayHandle.m_handle;
      auto derefdHandle = m_memPool.derefHandle(handle);
      for (Index_t i = 0; i < m_arrayHandle.GetCount(); ++i) {
        (derefdHandle[i]).~T();
      }

      arena.FreeFromArena(reinterpret_cast<std::byte *>(derefdHandle));
    }

    const ArrayHandle<T> &m_arrayHandle;
    PortableMemPool &m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  void DeallocArrayImpl(const ArrayHandle<T> &arrayHandle,
                        Arena<allocSizes, totalSizes> &...arenas) {
    WithArena<T>(MatchArenaOffset<T>(
                     *this, arrayHandle.m_handle.GetDistFromMemPoolBase()),
                 DeallocArrayHandler<T>(arrayHandle, *this), arenas...);
  }

  template <Index_t allocSize, Index_t totalSize, Index_t... allocSizes,
            Index_t... totalSizes>
  Index_t
  GetTotalAllocationCountImpl(Arena<allocSize, totalSize> &arena,
                              Arena<allocSizes, totalSizes> &...arenas) {
    Index_t totalAllocCount = 0;

    cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                         cl::sycl::memory_scope::device,
                         cl::sycl::access::address_space::global_space>
        totalAllocations(arena.m_totalAllocations);
    totalAllocCount += totalAllocations.load();
    totalAllocCount += GetTotalAllocationCountImpl(arenas...);

    return totalAllocCount;
  }

  template <Index_t allocSize, Index_t totalSize>
  Index_t GetTotalAllocationCountImpl(Arena<allocSize, totalSize> &arena) {
    cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                         cl::sycl::memory_scope::device,
                         cl::sycl::access::address_space::global_space>
        totalAllocations(arena.m_totalAllocations);
    return totalAllocations.load();
  }

  template <class T, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  Index_t GetNumFreeImpl(Arena<allocSize, totalSize> &arena,
                         Arena<allocSizes, totalSizes> &...arenas) {
    if (allocSize >= sizeof(T)) {
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          totalAllocations(arena.m_totalAllocations);
      return (totalSize / allocSize) - totalAllocations.load();
    } else {
      return GetNumFreeImpl<T>(arenas...);
    }
  }

  template <class T, Index_t allocSize, Index_t totalSize>
  Index_t GetNumFreeImpl(Arena<allocSize, totalSize> &arena) {
    if (allocSize >= sizeof(T)) {
      cl::sycl::atomic_ref<Index_t, cl::sycl::memory_order::seq_cst,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>
          totalAllocations(arena.m_totalAllocations);
      return (totalSize / allocSize) - totalAllocations.load();
    }
    return 0;
  }

#define BIN_SIZE 16777216
  Arena<sizeof(int) * 8, BIN_SIZE * 16> m_smallBin;
  Arena<sizeof(int) * 64, BIN_SIZE * 16> m_mediumBin;
  Arena<sizeof(int) * 16384, BIN_SIZE * 16> m_largeBin;
  Arena<sizeof(int) * 2097152, BIN_SIZE> m_extraLargeBin;
#undef BIN_SIZE
};
static_assert(sizeof(PortableMemPool) < std::numeric_limits<Index_t>::max(),
              "All pointer diffs need to be representable by Index_t");
} // namespace FunGPU
