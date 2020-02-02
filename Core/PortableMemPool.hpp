#pragma once

#include "Types.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/multi_ptr.hpp>
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

    // Do not free the handle returned from this function, that will break the
    // allocator.
    Handle<T> ElementHandle(const Index_t elemIdx) {
      return Handle<T>(m_handle.GetAllocIndex() + elemIdx * sizeof(T),
                       m_handle.GetAllocSize());
    }

    bool operator==(const ArrayHandle<T> &other) {
      return m_handle == other.m_handle && m_count == other.m_count;
    }

    bool operator!=(const ArrayHandle<T> &other) { return !(*this == other); }

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

    auto *derefdAllocd = derefHandle(handle);
    // invoke T's constructor via placement new on allocated bytes
    auto allocdT = new (derefdAllocd) T(args...);

    using SetHandleFunctor_t = typename std::conditional<
        std::is_base_of<EnableHandleFromThis<T>, T>::value, SetHandleReal<T>,
        SetHandleNoOp<T>>::type;
    SetHandleFunctor_t::SetHandle(*derefdAllocd, handle);

    return handle;
  }

  template<class T> bool TryReserve() {
    return TryReserveImpl<T>(m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
  }

  template <class T>
  ArrayHandle<T> AllocArray(const Index_t arraySize,
                            const T &initialValue = T()) {
    return AllocArrayImpl<T>(arraySize, initialValue, m_smallBin, m_mediumBin,
                             m_largeBin, m_extraLargeBin);
  }

  template<class T> bool TryReserveArray(const Index_t arraySize) {
    return TryReserveArrayImpl<T>(arraySize, m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
  }

  template <class T> void Dealloc(const Handle<T> &handle) {
    DeallocImpl(handle, m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
  }

  template <class T> void DeallocArray(const ArrayHandle<T> &arrayHandle) {
    DeallocArrayImpl(arrayHandle, m_smallBin, m_mediumBin, m_largeBin,
                     m_extraLargeBin);
  }

  void ClearReservations() {
    ClearReservationsImpl(m_smallBin, m_mediumBin, m_largeBin, m_extraLargeBin);
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
        m_allocBeginIndices[i] = i * AllocSize;
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

    void FreeFromArena(const Index_t allocIdx) {
      cl::sycl::atomic<Index_t> allocdBlockBegin(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_allocdBlockBegin)));
      const auto freeDst =
          allocdBlockBegin.fetch_add(1) % m_allocBeginIndices.size();
      m_allocBeginIndices[freeDst] = allocIdx;

      cl::sycl::atomic<Index_t> allocCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_totalAllocations)));
      allocCount.fetch_sub(1);
    }

    bool TryReserve() {
      cl::sycl::atomic<Index_t> reservedAllocCount((cl::sycl::multi_ptr<Index_t, cl::sycl::access::address_space::global_space>(&m_reservedAllocations)));
      const auto prevReservedCount = reservedAllocCount.fetch_add(1);
      // Implicit in this math is that no one is concurrently allocating and trying to reserve. Should have a separate reserve and alloc pass.
      return prevReservedCount + cl::sycl::atomic<Index_t>((cl::sycl::multi_ptr<Index_t, cl::sycl::access::address_space::global_space>(&m_totalAllocations))).load() < TotalNodes;
    }

    void ClearReserved() {
      cl::sycl::atomic<Index_t> reservedAllocCount(
          (cl::sycl::multi_ptr<Index_t,
                               cl::sycl::access::address_space::global_space>(
              &m_reservedAllocations)));
      reservedAllocCount.store(0);
    }

    unsigned char *GetBytes(const Index_t allocIdx) {
      return &m_storage[allocIdx];
    }
    alignas(AllocSize) unsigned char m_storage[TotalBytes];
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
  decltype(auto) WithArena(Pred&& pred, CB&& cb, Arena<allocSize, totalSize>& arena) {
    if (pred(arena)) {
      return cb(arena);
    }
    return cb();
  }

  template <class T, class Pred, class CB, Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  decltype(auto) WithArena(Pred&& pred, CB&& cb, Arena<allocSize, totalSize> &arena,
                      Arena<allocSizes, totalSizes> &... arenas) {
    if (pred(arena)) {
      return cb(arena);
    }
    return WithArena<T, Pred, CB>(std::forward<Pred>(pred), std::forward<CB>(cb),
                                 arenas...);
  }

  template<class T>
  struct AllocPred {
    template<Index_t allocSize, Index_t totalSize>
    constexpr bool operator()(const Arena<allocSize, totalSize>&) {
      return sizeof(T) <= allocSize;
    }
  };

  template<class T>
  struct AllocHandler {
    Handle<T> operator()() { return Handle<T>(); }
    template <class DispatchedArena>
    Handle<T> operator()(DispatchedArena &arena) {
      const auto allocdIdx = arena.AllocFromArena();
      if (allocdIdx == std::numeric_limits<Index_t>::max()) {
        return Handle<T>();
      }

      const Handle<T> resultHandle(
          allocdIdx, std::remove_reference<decltype(arena)>::type::AllocSize);
      return resultHandle;
    }
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  Handle<T> AllocImpl(Arena<allocSizes, totalSizes> &... arenas) {
    return WithArena<T>(AllocPred<T>(), AllocHandler<T>(), arenas...);
  }

  struct TryReserveHandler {
    bool operator()() { return false; }
    template <class DispatchedArena> bool operator()(DispatchedArena &arena) {
      return arena.TryReserve();
    }
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  bool TryReserveImpl(Arena<allocSizes, totalSizes>&... arenas) {
    return WithArena<T>(AllocPred<T>(), TryReserveHandler(), arenas...);
  }

  template<class T>
  struct AllocArrayPred {
    AllocArrayPred(const Index_t arraySize) : m_arraySize(arraySize) {}

    template<Index_t allocSize, Index_t totalSize>
    bool operator()(const Arena<allocSize, totalSize>& arena) {
      return allocSize >= sizeof(T) * m_arraySize;
    }

    const Index_t m_arraySize;
  };

  template<class T>
  struct AllocArrayHandler {
    AllocArrayHandler(const Index_t arraySize, const T& initialValue,
                      PortableMemPool& memPool) : m_arraySize(arraySize), m_initialValue(initialValue),
                                                                           m_memPool(memPool) {}

    ArrayHandle<T> operator()() {
      return ArrayHandle<T>();
    }

    template<class DispatchedArena>
    ArrayHandle<T> operator()(DispatchedArena& arena) {
      const auto allocdIdx = arena.AllocFromArena();
      if (allocdIdx == std::numeric_limits<Index_t>::max()) {
        return ArrayHandle<T>();
      }
      const ArrayHandle<T> resultHandle(
          allocdIdx, std::remove_reference<decltype(arena)>::type::AllocSize,
          m_arraySize);

      auto tAlignedValues = m_memPool.derefHandle(resultHandle);
      for (Index_t i = 0; i < m_arraySize; ++i) {
        tAlignedValues[i] = *(new (tAlignedValues + i) T(m_initialValue));
      }

      return resultHandle;
    }

    const Index_t m_arraySize;
    const T& m_initialValue;
    PortableMemPool& m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  ArrayHandle<T> AllocArrayImpl(const Index_t arraySize, const T &initialValue,
                                Arena<allocSizes, totalSizes> &... arenas) {
    return WithArena<T>(AllocArrayPred<T>(arraySize), AllocArrayHandler<T>(arraySize, initialValue, *this), arenas...);
  }

  struct ReserveArrayHandler {
    bool operator()() { return false; }
    template <class DispatchedArena> bool operator()(DispatchedArena &arena) {
      return arena.TryReserve();
    }
  };
  
  template<class T, Index_t... allocSizes, Index_t... totalSizes>
  bool TryReserveArrayImpl(const Index_t arraySize, Arena<allocSizes, totalSizes> &... arenas) {
    return WithArena<T>(AllocArrayPred<T>(arraySize), ReserveArrayHandler(), arenas...);
  }

  template<class T>
  struct MatchAllocSizePred {
    MatchAllocSizePred(const Index_t allocSize) : m_allocSize(allocSize) {}

    template<Index_t allocSize, Index_t totalSize>
    bool operator()(const Arena<allocSize, totalSize>&) {
      return allocSize == m_allocSize;
    }

    Index_t m_allocSize;
  };

  template<class T>
  struct DeallocHandler {
    DeallocHandler(const Handle<T>& handle, PortableMemPool& memPool) : m_handle(handle),
                                                                        m_memPool(memPool) {}
    void operator()() {}

    template<class DispatchedArena>
    void operator()(DispatchedArena& arena) {
      auto derefedForHandle = m_memPool.derefHandle(m_handle);
      // Explicitly call destructor
      derefedForHandle->~T();
      arena.FreeFromArena(m_handle.GetAllocIndex());
    }

    Handle<T> m_handle;
    PortableMemPool& m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  void DeallocImpl(const Handle<T> &handle, Arena<allocSizes, totalSizes> &... arenas) {
    WithArena<T>(MatchAllocSizePred<T>(handle.GetAllocSize()), DeallocHandler<T>(handle, *this), arenas...);
  }

  template<class T>
  struct DeallocArrayHandler {
    DeallocArrayHandler(const ArrayHandle<T>& arrayHandle, PortableMemPool& memPool) :
      m_arrayHandle(arrayHandle), m_memPool(memPool) {}

    void operator()() {}

    template<class DispatchedArena>
    void operator()(DispatchedArena& arena) {
      const auto &handle = m_arrayHandle.m_handle;
      auto derefdHandle = m_memPool.derefHandle(handle);
      for (Index_t i = 0; i < m_arrayHandle.GetCount(); ++i) {
        (derefdHandle[i]).~T();
      }

      arena.FreeFromArena(handle.GetAllocIndex());
    }

    const ArrayHandle<T>& m_arrayHandle;
    PortableMemPool& m_memPool;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  void DeallocArrayImpl(const ArrayHandle<T> &arrayHandle,
                        Arena<allocSizes, totalSizes> &... arenas) {
    WithArena<T>(MatchAllocSizePred<T>(arrayHandle.m_handle.GetAllocSize()),
              DeallocArrayHandler<T>(arrayHandle, *this), arenas...);
  }

  template <Index_t allocSize, Index_t totalSize>
  void ClearReservationsImpl(Arena<allocSize, totalSize>& arena) {
    arena.ClearReserved();
  }

  template <Index_t allocSize, Index_t totalSize,
            Index_t... allocSizes, Index_t... totalSizes>
  void ClearReservationsImpl(Arena<allocSize, totalSize>& arena,
                         Arena<allocSizes, totalSizes>&... arenas) {
    arena.ClearReserved();
    ClearReservationsImpl(arenas...);
  }

  template<class T>
  struct DerefHandleHandler {
    DerefHandleHandler(const Index_t allocIndex) : m_allocIndex(allocIndex) {}

    T* operator()() {
      return nullptr;
    }

    template<class DispatchedArena>
    T* operator()(DispatchedArena& arena) {
      return reinterpret_cast<T*>(arena.GetBytes(m_allocIndex));
    }

    const Index_t m_allocIndex;
  };

  template <class T, Index_t... allocSizes, Index_t... totalSizes>
  T *derefHandleImpl(const Handle<T> &handle,
                     Arena<allocSizes, totalSizes> &... arenas) {
    return WithArena<T>(MatchAllocSizePred<T>(handle.GetAllocSize()),
                        DerefHandleHandler<T>(handle.GetAllocIndex()), arenas...);
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
  Arena<sizeof(int) * 8, binSize * 16> m_mediumBin;
  Arena<sizeof(int) * 128, binSize * 16> m_largeBin;
  Arena<sizeof(int) * 2097152, binSize> m_extraLargeBin;
};
} // namespace FunGPU
