#include "PortableMemPool.h"
#include <algorithm>

namespace FunGPU
{
	template<>
	cl::sycl::accessor<PortableMemPool::Arena, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> PortableMemPool::GetArenasAccessorForTarget()
	{
		return m_arenasAccessors.GetDeviceAcc();
	}

	template<>
	cl::sycl::accessor<PortableMemPool::Arena, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> PortableMemPool::GetArenasAccessorForTarget()
	{
		return m_arenasAccessors.GetHostAcc();
	}

	PortableMemPool::PortableMemPool(std::vector<std::pair<size_t, size_t>> arenaSizesAndBytesPerArena,
		cl::sycl::handler* handler) :
		m_arenas(cl::sycl::range<1>(arenaSizesAndBytesPerArena.size())),
		m_arenasAccessors(m_arenas),
		m_isOnDevice(false)
	{
		std::sort(arenaSizesAndBytesPerArena.begin(), arenaSizesAndBytesPerArena.end());

		auto arenasHostAcc = m_arenas.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(handler);
		for (auto i = 0; i < arenaSizesAndBytesPerArena.size(); ++i)
		{
			const auto& arenaDesc = arenaSizesAndBytesPerArena[i];
			arenasHostAcc[i] = Arena(arenaDesc.first, arenaDesc.second, handler);
		}
	}

	void PortableMemPool::ConfigureForDeviceAccess(cl::sycl::handler& handler)
	{
		if (!m_isOnDevice)
		{
			m_arenasAccessors.ConfigureForDevice(handler);
			m_isOnDevice = true;
		}
	}

	void PortableMemPool::ConfigureForHostAccess(cl::sycl::handler& handler)
	{
		if (m_isOnDevice)
		{
			m_arenasAccessors.ConfigureForHost(handler);
			m_isOnDevice = false;
		}
	}

	PortableMemPool::Arena::Arena(const size_t sizeOfEachAlloc, const size_t totalBytes,
		cl::sycl::handler* handler):
		m_sizeOfEachUnit(sizeOfEachAlloc), m_bytes(cl::sycl::range<1>(totalBytes)),
		m_listNodeStorage(cl::sycl::range<1>(totalBytes / sizeOfEachAlloc)),
	{
		if (totalBytes % m_sizeOfEachUnit != 0)
		{
			throw std::invalid_argument("Expected total bytes to be a multiple of allocation size");
		}

		auto listNodeAcc = m_listNodeStorage.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(*m_handler);

		for (size_t i = 0; i < m_listNodeStorage.get_size(); ++i)
		{
			auto& listNode = listNodeAcc[i];
			listNode = ListNode();
			listNode.m_beginIndex = i * m_sizeOfEachUnit;
			listNode.m_indexInStorage = i;
			listNode.m_nextNode = i + 1;
		}

		m_freeListHead = listNodeAcc[0];
		m_freeListTail.m_beginIndex = m_bytes.get_size() + 1;
		m_freeListTail.m_nextNode = m_listNodeStorage.get_size() + 1;
	}

	PortableMemPool::Arena::Arena(const Arena& other): Arena(other.m_sizeOfEachUnit, other.m_listNodeStorage.get_size() * other.m_sizeOfEachUnit, other.m_handler)
	{
	}

	template<cl::sycl::access::target target>
	PortableMemPool::ListNode PortableMemPool::Arena::AllocFromArena()
	{
		SpinLockGuard guard(m_lock);

		if (m_freeListHead == m_freeListTail)
		{
			throw std::runtime_error("Out of memory");
		}
		
		const auto result = m_freeListHead;
		const auto nextListIndex = m_freeListHead.m_nextNode;

		auto listNodeStorageAcc = m_listNodeStorage.get_access<cl::sycl::access::mode::read_write, target>(*m_handler);
		if (nextListIndex >= m_listNodeStorage.get_size())
		{
			m_freeListHead = m_freeListTail;
		}
		else
		{
			m_freeListHead = listNodeStorageAcc[nextListIndex];
		}

		++m_totalAllocations;

		return result;
	}

	template<cl::sycl::access::target target>
	void PortableMemPool::Arena::FreeFromArena(const size_t indexInStorage)
	{
		SpinLockGuard guard(m_lock);

		auto listNodeStorageAcc = m_listNodeStorage.get_access<cl::sycl::access::mode::read_write, target>(*m_handler);
		listNodeStorageAcc[indexInStorage].m_nextNode = m_freeListHead.m_indexInStorage;
		m_freeListHead = listNodeStorageAcc[indexInStorage];

		--m_totalAllocations;
	}

	template<cl::sycl::access::target target>
	unsigned char* PortableMemPool::Arena::GetBytes(const size_t allocIndex)
	{
		auto listNodeStorageAcc = m_listNodeStorage.get_access<cl::sycl::access::mode::read_write, target>(*m_handler);
		const auto& listNodeForAllocIndex = listNodeStorageAcc[allocIndex];
		return &m_bytes.at(listNodeForAllocIndex.m_beginIndex);
	}

	template<cl::sycl::access::target target>
	size_t PortableMemPool::FindArenaForAllocSize(const size_t allocSize) const
	{
		auto arenasAcc = m_arenas.get_access<cl::sycl::access::mode::read_write, target>(*m_handler);
		auto iter = std::lower_bound(m_arenas->begin(), m_arenas->end(), allocSize, 
			[](const Arena& a1, const size_t theAllocSize) -> bool{
				return a1.m_sizeOfEachUnit < theAllocSize;
			}
		);
		if (iter == m_arenas->end())
		{
			throw std::invalid_argument("Tried to alloc a size larger than largest pool in mem pool");
		}
		return std::distance(m_arenas->begin(), iter);
	}

	template<cl::sycl::access::target target>
	size_t PortableMemPool::GetTotalAllocationCount()
	{
		auto arenaAcc = m_arenas.get_access<cl::sycl::access::mode::read_write, target>(*m_handler);
		size_t totalAllocationCount = 0;
		for (size_t i = 0; i < m_arenas.get_size(); ++i)
		{
			auto& arena = arenaAcc[i];
			SpinLockGuard guard(arena.m_lock);
			totalAllocationCount += arena.m_totalAllocations;
		}

		return totalAllocationCount;
	}
}
