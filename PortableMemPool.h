#pragma once

#include <vector>
#include <utility>
#include <type_traits>
#include "SpinLock.h"

namespace FunGPU
{
	/**
	* Arena-based memory pool that can be moved around without invalidating references to objects in it.
	* Required to alloc compiled nodes on host and reference them on device.
	*/
	class PortableMemPool
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

			struct SetHandleNoOp
			{
				SetHandleNoOp(T*) {}
				void SetHandle(const Handle<T>&) {}
			};

			struct SetHandleReal
			{
				SetHandleReal(GetHandleFromThis<T>* target) : m_target(target) {}
				void SetHandle(const Handle<T>& handle)
				{
					m_target->m_handle = handle;
				}

				GetHandleFromThis<T>* m_target;
			};

			using SetHandleFunctor_t = typename std::conditional<std::is_base_of<GetHandleFromThis<T>, T>::value, SetHandleReal, SetHandleNoOp>::type;
			SetHandleFunctor_t setHandleFunctor(allocdT);
			setHandleFunctor.SetHandle(resultHandle);
			
			return resultHandle;
		}

		template<class T>
		void Dealloc(const Handle<T>& handle)
		{
			auto derefedForHandle = derefHandle(resultHandle);
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
		};
		
		size_t FindArenaForAllocSize(const size_t allocSize) const;

		std::vector<Arena> m_arenas;
	};
}
