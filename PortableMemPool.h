#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <type_traits>
#include "SpinLock.h"
#include <atomic>
#include <SYCL\sycl.hpp>
#include "Sycl1DBufferAccessors.hpp"

namespace FunGPU
{
	/**
	* Arena-based memory pool that can be moved around without invalidating references to objects in it.
	* Required to alloc compiled nodes on host and reference them on device.
	*/
	class PortableMemPool
	{
	public:
		using DeviceAccessor_t = cl::sycl::accessor<PortableMemPool, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>;

		template<class T>
		class Handle
		{
		public:
			Handle() : m_arenaIndex(std::numeric_limits<size_t>::max()), 
				m_allocIndex(std::numeric_limits<size_t>::max()), 
				m_allocSize(std::numeric_limits<size_t>::max()) {}
			Handle(const size_t arenaIndex, const size_t allocIndex, const size_t allocSize) :
				m_arenaIndex(arenaIndex), m_allocIndex(allocIndex), m_allocSize(allocSize) {}

			template<class OtherT>
			Handle(const Handle<OtherT>& other)
			{
				*this = other;
			}

			size_t GetArenaIndex() const
			{
				return m_arenaIndex;
			}

			size_t GetAllocIndex() const
			{
				return m_allocIndex;
			}

			size_t GetAllocSize() const
			{
				return m_allocSize;
			}
			bool operator==(const Handle& other) const 
			{
				return m_arenaIndex == other.m_arenaIndex &&
					m_allocIndex == other.m_allocIndex &&
					m_allocSize == other.m_allocSize;
			}
			bool operator!=(const Handle& other) const
			{
				return !(*this == other);
			}

			template<class OtherT>
			void operator=(const Handle<OtherT>& other)
			{
				static_assert(std::is_base_of<T, OtherT>::value || std::is_base_of<OtherT, T>::value,
					"Cannot assign handle to handle of unrelated type");
				m_arenaIndex = other.GetArenaIndex();
				m_allocIndex = other.GetAllocIndex();
				m_allocSize = other.GetAllocSize();
			}
			
		private:
			size_t m_arenaIndex;
			size_t m_allocIndex;
			size_t m_allocSize;
		};

		template<class T>
		class ArrayHandle
		{
			friend class PortableMemPool;
		public:
			ArrayHandle() : m_count(std::numeric_limits<size_t>::max()) {}
			ArrayHandle(const size_t arenaIndex, const size_t allocIndex, const size_t allocSize, const size_t count): 
				m_handle(arenaIndex, allocIndex, allocSize), m_count(count) {}

			size_t GetCount() const
			{
				return m_count;
			}

		private:
			Handle<T> m_handle;
			size_t m_count;
		};

		template<class T>
		class SharedHandle
		{
			friend class PortableMemPool;
		public:
			SharedHandle() {}

			SharedHandle(const Handle<T>& handle, PortableMemPool* pool) : m_wrappedHandle(handle),
				m_pool(pool), 
				m_refCountHandle(pool->Alloc<std::atomic<size_t>>(1))
			{
			}

			SharedHandle(const SharedHandle<T>& other):
				m_wrappedHandle(other.m_wrappedHandle),
				m_pool(other.m_pool), m_refCountHandle(other.m_refCountHandle)
			{
				IncrementRefCount();
			}

			~SharedHandle()
			{
				DecrementRefCount();
			}

			void SetPool(PortableMemPool* pool)
			{
				m_pool = pool;
			}

			void Detach()
			{
				m_isAttached = false;
			}

			SharedHandle<T> operator=(const SharedHandle<T>& other)
			{
				DecrementRefCount();

				m_wrappedHandle = other.m_wrappedHandle;
				m_pool = other.m_pool;
				m_refCountHandle = other.m_refCountHandle;

				IncrementRefCount();

				return *this;
			}

			bool operator==(const SharedHandle<T>& other) const
			{
				return m_wrappedHandle == other.m_wrappedHandle;
			}
			bool operator!=(const SharedHandle<T>& other) const
			{
				return !(*this == other);
			}

		private:

			void IncrementRefCount()
			{
				if (m_wrappedHandle != Handle<T>())
				{
					auto derefdRefCount = m_pool->derefHandle(m_refCountHandle);
					++(*derefdRefCount);
				}
			}

			void DecrementRefCount()
			{
				if (m_isAttached && m_wrappedHandle != Handle<T>())
				{
					auto derefdRefCount = m_pool->derefHandle(m_refCountHandle);
					if (--(*derefdRefCount) == 0)
					{
						m_pool->Dealloc(m_refCountHandle);
						m_pool->Dealloc(m_wrappedHandle);
					}
				}
			}

			Handle<T> m_wrappedHandle;
			PortableMemPool* m_pool;
			Handle<std::atomic<size_t>> m_refCountHandle;

			bool m_isAttached = true;
		};

		// Implementers must define a public m_handle member. 
		// Can't define it for you here because then your class would not be a standard layout class :(
		template<class T>
		class EnableSharedHandleFromThis
		{
			//SharedHandle<T> m_handle;
		};

		PortableMemPool(std::vector<std::pair<size_t, size_t>> arenaSizesAndBytesPerArena, cl::sycl::handler* handler);
		
		void ConfigureForHostAccess(cl::sycl::handler& handler);
		void ConfigureForDeviceAccess(cl::sycl::handler& handler);

		template<cl::sycl::access::target target, class T, class ...Args_t>
		Handle<T> Alloc(const Args_t&... args)
		{
			auto accessorForTarget = GetArenasAccessorForTarget<target>();
			return AllocImpl<decltype<accessorForTarget>, T, Args_t...>(accessorForTarget, std::forward(args...));
		}

		template<cl::sycl::access::target target, class T>
		ArrayHandle<T> AllocArray(const size_t arraySize, const T& initialValue = T())
		{
			auto accessorForTarget = GetArenasAccessorForTarget<target>();
			return AllocArrayImpl<decltype<accessorForTarget>, T, Args_t...>(arraySize, std::forward(initialValue));
		}
		
		// TODO make the other alloc / dealloc routines follow per-target pattern above, and do the same with arena.
		template<class T, class ...Args_t>
		SharedHandle<T> AllocShared(const Args_t&... args)
		{
			SharedHandle<T> result(Alloc<T, Args_t...>(args...), this);

			using SetHandleFunctor_t = typename std::conditional<std::is_base_of<EnableSharedHandleFromThis<T>, T>::value, SetSharedHandleReal<T>, SetHandleNoOp<T>>::type;
			auto derefedAllocd = derefHandle(result);
			SetHandleFunctor_t setHandleFunctor(derefedAllocd);
			setHandleFunctor.SetHandle(result);
			// TODO move the decrement ref count call into setHandleFunctor.SetHandle and only do it in SetSharedHandleReal
			result.DecrementRefCount();

			return result;
		}

		template<class T>
		void Dealloc(const Handle<T>& handle)
		{
			auto derefedForHandle = derefHandle(handle);
			// Explicitly call destructor
			derefedForHandle->~T();
			auto& arena = m_arenas->at(handle.GetArenaIndex());
			arena.FreeFromArena(handle.GetAllocIndex());
		}

		template<class T>
		void DeallocArray(const ArrayHandle<T>& arrayHandle)
		{
			const auto& handle = arrayHandle.m_handle;
			auto derefdHandle = derefHandle(handle);
			for (size_t i = 0; i < arrayHandle.GetCount(); ++i)
			{
				(derefdHandle[i]).~T();
			}

			auto& arena = m_arenas->at(handle.GetArenaIndex());
			arena.FreeFromArena(handle.GetAllocIndex());
		}

		template<class T>
		T* derefHandle(const Handle<T>& handle)
		{
			auto& arena = m_arenas->at(handle.GetArenaIndex());
			auto bytesForIndex = arena.GetBytes(handle.GetAllocIndex());
			return reinterpret_cast<T*>(bytesForIndex);
		}

		template<class T>
		T* derefHandle(const SharedHandle<T>& handle)
		{
			return derefHandle(handle.m_wrappedHandle);
		}

		template<class T>
		T* derefHandle(const ArrayHandle<T>& handle)
		{
			return derefHandle(handle.m_handle);
		}

		template<cl::sycl::access::target target>
		size_t GetTotalAllocationCount();

	private:
		struct ListNode
		{
			ListNode(const size_t beginIndex) : m_beginIndex(beginIndex) {}
			ListNode() : m_beginIndex(0), m_nextNode(0) {}
			bool operator==(const ListNode& other)
			{
				return m_beginIndex == other.m_beginIndex &&
					m_nextNode == other.m_nextNode &&
					m_indexInStorage == other.m_indexInStorage;
			}

			size_t m_beginIndex;
			size_t m_nextNode;
			size_t m_indexInStorage;
		};

		struct Arena
		{
			Arena(const size_t sizeOfEachAlloc, const size_t totalBytes,
				cl::sycl::handler* handler);
			Arena(const Arena& other);

			ListNode AllocFromArena();

			void FreeFromArena(const size_t allocIndex);

			unsigned char* GetBytes(const size_t allocIndex);

			const size_t m_sizeOfEachUnit;
			
			cl::sycl::buffer<unsigned char, 1> m_bytes;
			cl::sycl::buffer<ListNode, 1> m_listNodeStorage;

			ListNode m_freeListHead;
			ListNode m_freeListTail;
			SpinLock m_lock;

			size_t m_totalAllocations = 0;
		};
		
		template<class T>
		struct SetHandleNoOp
		{
			SetHandleNoOp(T*) {}
			void SetHandle(const Handle<T>&) {}
		};

		template<class T>
		struct SetSharedHandleReal
		{
			SetSharedHandleReal(EnableSharedHandleFromThis<T>* target) : m_target(target) {}
			void SetHandle(const SharedHandle<T>& handle)
			{
				auto targetAsOriginalClass = static_cast<T*>(m_target);
				targetAsOriginalClass->m_handle = handle;
				targetAsOriginalClass->m_handle.Detach();
			}

			EnableSharedHandleFromThis<T>* m_target;
		};

		template<cl::sycl::access::target target>
		cl::sycl::accessor<Arena, 1, cl::sycl::access::mode::read_write, target> GetArenasAccessorForTarget();

		template<class Accessor_t, class T, class ...Args_t>
		Handle<T> AllocImpl(const Accessor_t& arenasAcc, const Args_t&... args)
		{
			const auto arenaIndex = FindArenaForAllocSize(sizeof(T));
			auto& arena = arenasAcc[arenaIndex];

			const auto allocdListNode = arena.AllocFromArena();
			const Handle<T> resultHandle(arenaIndex, allocdListNode.m_indexInStorage, arena.m_sizeOfEachUnit);
			auto bytesForHandle = reinterpret_cast<unsigned char*>(derefHandle(resultHandle));

			// invoke T's constructor via placement new on allocated bytes
			auto allocdT = new (bytesForHandle) T(args...);

			return resultHandle;
		}

		template<class Accessor_t, class T>
		ArrayHandle<T> AllocArrayImpl(const Accessor_t& arenasAcc, const size_t arraySize, const T& initialValue = T())
		{
			const auto arenaIndex = FindArenaForAllocSize(sizeof(T) * arraySize);
			auto& arena = arenasAcc[arenaIndex];

			const auto allocdListNode = arena.AllocFromArena();
			const ArrayHandle<T> resultHandle(arenaIndex, allocdListNode.m_indexInStorage, arena.m_sizeOfEachUnit, arraySize);

			auto tAlignedValues = derefHandle(resultHandle);
			for (size_t i = 0; i < arraySize; ++i)
			{
				tAlignedValues[i] = *(new (tAlignedValues + i) T(initialValue));
			}

			return resultHandle;
		}

		size_t FindArenaForAllocSize(const size_t allocSize) const;

		cl::sycl::buffer<Arena, 1> m_arenas;
		Sycl1DBufferWithAccessors<cl::sycl::buffer<Arena, 1>> m_arenasAccessors;

		bool m_isOnDevice;
	};
}
