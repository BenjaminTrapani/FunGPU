#include "PortableMemPool.h"
#include <algorithm>

namespace FunGPU
{
	PortableMemPool::PortableMemPool(std::vector<std::pair<size_t, size_t>> arenaSizesAndBytesPerArena)
	{
		std::sort(arenaSizesAndBytesPerArena.begin(), arenaSizesAndBytesPerArena.end());
		m_arenas.reserve(arenaSizesAndBytesPerArena.size());
		for (const auto& arenaDesc : arenaSizesAndBytesPerArena)
		{
			m_arenas.push_back(Arena(arenaDesc.first, arenaDesc.second));
		}
	}

	PortableMemPool::Arena::Arena(const size_t sizeOfEachAlloc, const size_t totalBytes):
		m_sizeOfEachUnit(sizeOfEachAlloc), m_bytes(totalBytes)
	{
		if (totalBytes % m_sizeOfEachUnit != 0)
		{
			throw std::invalid_argument("Expected total bytes to be a multiple of allocation size");
		}
		const auto numberOfListNodesRequired = totalBytes / m_sizeOfEachUnit;
		m_listNodeStorage = std::vector<ListNode>(numberOfListNodesRequired);

		for (size_t i = 0; i < m_listNodeStorage.size(); ++i)
		{
			auto& listNode = m_listNodeStorage[i];
			listNode.m_beginIndex = i * m_sizeOfEachUnit;
			listNode.m_indexInStorage = i;
			listNode.m_nextNode = i + 1;
		}

		m_freeListHead = m_listNodeStorage.at(0);
		m_freeListTail.m_beginIndex = m_bytes.size() + 1;
		m_freeListTail.m_nextNode = m_listNodeStorage.size() + 1;
	}

	PortableMemPool::ListNode PortableMemPool::Arena::AllocFromArena()
	{
		SpinLockGuard guard(m_lock);

		if (m_freeListHead == m_freeListTail)
		{
			throw std::runtime_error("Out of memory");
		}
		
		const auto result = m_freeListHead;
		const auto nextListIndex = m_freeListHead.m_nextNode;
		if (nextListIndex >= m_listNodeStorage.size())
		{
			m_freeListHead = m_freeListTail;
		}
		else
		{
			m_freeListHead = m_listNodeStorage.at(nextListIndex);
		}

		++m_totalAllocations;

		return result;
	}

	void PortableMemPool::Arena::FreeFromArena(const size_t indexInStorage)
	{
		SpinLockGuard guard(m_lock);

		m_listNodeStorage.at(indexInStorage).m_nextNode = m_freeListHead.m_indexInStorage;
		m_freeListHead = m_listNodeStorage.at(indexInStorage);

		--m_totalAllocations;
	}

	unsigned char* PortableMemPool::Arena::GetBytes(const size_t allocIndex)
	{
		const auto& listNodeForAllocIndex = m_listNodeStorage.at(allocIndex);
		return &m_bytes.at(listNodeForAllocIndex.m_beginIndex);
	}

	size_t PortableMemPool::FindArenaForAllocSize(const size_t allocSize) const
	{
		auto iter = std::lower_bound(m_arenas.begin(), m_arenas.end(), allocSize, 
			[](const Arena& a1, const size_t theAllocSize) -> bool{
				return a1.m_sizeOfEachUnit < theAllocSize;
			}
		);
		if (iter == m_arenas.end())
		{
			throw std::invalid_argument("Tried to alloc a size larger than largest pool in mem pool");
		}
		return std::distance(m_arenas.begin(), iter);
	}

	size_t PortableMemPool::GetTotalAllocationCount()
	{
		size_t totalAllocationCount = 0;
		for (auto& arena : m_arenas)
		{
			SpinLockGuard guard(arena.m_lock);
			totalAllocationCount += arena.m_totalAllocations;
		}

		return totalAllocationCount;
	}
}
