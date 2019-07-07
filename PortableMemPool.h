#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <type_traits>
#include "SpinLock.h"
#include <atomic>

namespace FunGPU
{
	/**
	* Arena-based memory pool that can be moved around without invalidating references to objects in it.
	* Required to alloc compiled nodes on host and reference them on device.
	*/
	class PortableMemPool : public std::enable_shared_from_this<PortableMemPool>
	{
	public:

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
				static_assert(std::is_base_of<T, OtherT>::value, 
					"Cannot assign handle to a handle to a type that is not a subclass of the destination handle's type");
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
		class SharedHandle : public Handle<T>
		{
			friend class PortableMemPool;
		public:
			SharedHandle() : Handle<T>() {}

			SharedHandle(const Handle<T>& handle, const std::shared_ptr<PortableMemPool>& pool) : Handle<T>(handle),
				m_pool(pool), 
				m_refCountHandle(pool->Alloc<std::atomic<size_t>>(1))
			{
			}

			SharedHandle(const SharedHandle<T>& other):
				Handle<T>(other),
				m_pool(other.m_pool), m_refCountHandle(other.m_refCountHandle)
			{
				IncrementRefCount();
			}

			~SharedHandle()
			{
				DecrementRefCount();
			}

			void SetPool(const std::shared_ptr<PortableMemPool>& pool)
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

				*(static_cast<Handle<T>*>(this)) = other;

				m_pool = other.m_pool;
				m_refCountHandle = other.m_refCountHandle;

				IncrementRefCount();

				return *this;
			}

		private:

			void IncrementRefCount()
			{
				if (*this != Handle<T>())
				{
					auto derefdRefCount = m_pool->derefHandle(m_refCountHandle);
					++(*derefdRefCount);
				}
			}

			void DecrementRefCount()
			{
				if (m_isAttached && *this != Handle<T>())
				{
					auto derefdRefCount = m_pool->derefHandle(m_refCountHandle);
					if (--(*derefdRefCount) == 0)
					{
						// Problem is double-free from shared_from_this code
						m_pool->Dealloc(m_refCountHandle);
						m_pool->Dealloc(*this);
					}
				}
			}

			std::shared_ptr<PortableMemPool> m_pool;
			Handle<std::atomic<size_t>> m_refCountHandle;
			bool m_isAttached = true;
		};

		template<class T>
		class GetHandleFromThis
		{
			friend class PortableMemPool;
		protected:
			Handle<T> GetHandle()
			{
				return m_handle;
			}
		private:
			Handle<T> m_handle;
		};

		template<class T>
		class GetSharedHandleFromThis
		{
			friend class PortableMemPool;
		protected:
			SharedHandle<T> GetHandle()
			{
				return m_handle;
			}
		private:
			SharedHandle<T> m_handle;
		};

		PortableMemPool(std::vector<std::pair<size_t, size_t>> arenaSizesAndBytesPerArena);
		
		template<class T, class ...Args_t>
		Handle<T> Alloc(const Args_t&... args)
		{
			const auto arenaIndex = FindArenaForAllocSize(sizeof(T));
			auto& arena = m_arenas.at(arenaIndex);
			const auto allocdListNode = arena.AllocFromArena();
			const Handle<T> resultHandle(arenaIndex, allocdListNode.m_indexInStorage, arena.m_sizeOfEachUnit);
			auto bytesForHandle = reinterpret_cast<unsigned char*>(derefHandle(resultHandle));
			
			// invoke T's constructor via placement new on allocated bytes
			auto allocdT = new (bytesForHandle) T(args...);

			using SetHandleFunctor_t = typename std::conditional<std::is_base_of<GetHandleFromThis<T>, T>::value, SetHandleReal<T>, SetHandleNoOp<T>>::type;
			SetHandleFunctor_t setHandleFunctor(allocdT);
			setHandleFunctor.SetHandle(resultHandle);
			
			return resultHandle;
		}

		template<class T, class ...Args_t>
		SharedHandle<T> AllocShared(const Args_t&... args)
		{
			SharedHandle<T> result(Alloc<T, Args_t...>(args...), shared_from_this());

			using SetHandleFunctor_t = typename std::conditional<std::is_base_of<GetSharedHandleFromThis<T>, T>::value, SetSharedHandleReal<T>, SetHandleNoOp<T>>::type;
			auto derefedAllocd = derefHandle(result);
			SetHandleFunctor_t setHandleFunctor(derefedAllocd);
			setHandleFunctor.SetHandle(result);
			result.DecrementRefCount();

			return result;
		}

		template<class T>
		void Dealloc(const Handle<T>& handle)
		{
			auto derefedForHandle = derefHandle(handle);
			// Explicitly call destructor
			derefedForHandle->~T();

			auto& arena = m_arenas.at(handle.GetArenaIndex());
			arena.FreeFromArena(handle.GetAllocIndex());
		}

		template<class T>
		T* derefHandle(const Handle<T>& handle)
		{
			auto& arena = m_arenas.at(handle.GetArenaIndex());
			auto bytesForIndex = arena.GetBytes(handle.GetAllocIndex());
			return reinterpret_cast<T*>(bytesForIndex);
		}

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
			Arena(const size_t sizeOfEachAlloc, const size_t totalBytes);
			ListNode AllocFromArena();
			void FreeFromArena(const size_t allocIndex);
			unsigned char* GetBytes(const size_t allocIndex);

			const size_t m_sizeOfEachUnit;
			std::vector<unsigned char> m_bytes;
			std::vector<ListNode> m_listNodeStorage;
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
		struct SetHandleReal
		{
			SetHandleReal(GetHandleFromThis<T>* target) : m_target(target) {}
			void SetHandle(const Handle<T>& handle)
			{
				m_target->m_handle = handle;
			}

			GetHandleFromThis<T>* m_target;
		};

		template<class T>
		struct SetSharedHandleReal
		{
			SetSharedHandleReal(GetSharedHandleFromThis<T>* target) : m_target(target) {}
			void SetHandle(const SharedHandle<T>& handle)
			{
				m_target->m_handle = handle;
				m_target->m_handle.Detach();
			}

			GetSharedHandleFromThis<T>* m_target;
		};

		size_t FindArenaForAllocSize(const size_t allocSize) const;

		std::vector<Arena> m_arenas;
	};
}
