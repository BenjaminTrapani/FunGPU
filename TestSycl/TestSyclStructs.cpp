#include "SYCL/sycl.hpp"
#include <vector>
#include <utility>
#include <iostream>
/*
struct ListNode
{
    int data = 0;
    int nextIdx = 0;
};

struct ArenaData {
    ListNode head;
    ListNode tail;
};

struct ArenaStorage
{
    cl::sycl::buffer<ListNode> m_listNodes;
    cl::sycl::buffer<unsigned char> m_bytes;
    cl::sycl::buffer<ArenaData> m_data;
    ArenaStorage(const size_t memSize, const size_t allocSize) : m_listNodes(cl::sycl::range<1>(memSize / allocSize)),
        m_bytes(memSize), m_data(cl::sycl::range<1>(1)) {}
};

struct ArenaStorageBlock
{
	std::vector<ArenaStorage> m_arenaStorage;

	ArenaStorageBlock(const std::vector<size_t, size_t> memSizeAllocSizes)
	{
		for (const auto& memSizeAllocSize : memSizeAllocSizes)
		{
			m_arenaStorage.push_back(ArenaStorage(memSizeAllocSize.first, memSizeAllocSize.second));
		}
	}
};

template<cl::sycl::access::target target>
struct Arena
{
    cl::sycl::accessor<ListNode, 1, cl::sycl::access::mode::read_write, target> m_listNodesAcc;
    cl::sycl::accessor<unsigned char, 1, cl::sycl::access::mode::read_write, target> m_bytesAcc;
    cl::sycl::accessor<ArenaData, 1, cl::sycl::access::mode::read_write, target> m_dataAcc;

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::global_buffer>>
	Arena(ArenaStorage& storage, cl::sycl::handler& cgh):
		m_listNodesAcc(storage.m_listNodes.get_access<cl::sycl::access::mode::read_write>()),
		m_bytesAcc(storage.m_bytes.get_access<cl::sycl::access::mode::read_write>()),
		m_dataAcc(storage.m_data.get_access<cl::sycl::access::mode::read_write>()) {
	}

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::host_buffer>>
    Arena(ArenaStorage& storage) :
		m_listNodesAcc(storage.m_listNodes.get_access<cl::sycl::access::mode::read_write>()),
		m_bytesAcc(storage.m_bytes.get_access<cl::sycl::access::mode::read_write>()),
		m_dataAcc(storage.m_data.get_access<cl::sycl::access::mode::read_write>()) {
	}

	Arena() {}
};

template<cl::sycl::access::target target>
struct MemPoolVecStorage
{
	std::vector<Arena<target>> m_arenas;

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::host_buffer>>
	MemPoolVecStorage(ArenaStorageBlock& storage)
	{
		for (auto& arenaStorage : storage.m_arenaStorage)
		{
			m_arenas.push_back(Arena<target>(arenaStorage));
		}
	}

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::global_buffer>>
	MemPoolVecStorage(std::vector<ArenaStorage>& storage, cl::sycl::handler& cgh)
	{
		for (auto& arenaStorage : storage.m_arenaStorage)
		{
			m_arenas.push_back(Arena<target>(arenaStorage, cgh));
		}
	}
};

template<cl::sycl::access::target target>
struct MemPoolStorage
{
	cl::sycl::buffer<Arena<target>> m_arenas;
	
	MemPoolStorage(MemPoolVecStorage& storage) : m_arenas(storage.m_arenas.data(), cl::sycl::range<1>(storage.m_arenas.size())
	{
	}
};

template<cl::sycl::access::target target>
struct MemPool
{
	cl::sycl::accessor<Arena<target>, 1, cl::sycl::access::mode::read_write, target> m_arenasAcc;

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::host_buffer>>
	MemPool(MemPoolStorage<target>& storage) : m_arenasAcc(storage.m_arenas.get_access<cl::sycl::access:::mode::read_write, target>()) {}

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::global_buffer>>
	MemPool(MemPoolStorage<target>& storage, cl::sycl::handler& cgh) : m_arenasAcc(storage.m_arenas.get_access<cl::sycl::access:::mode::read_write, target>(cgh)) {}

	void FakeAlloc() const
	{
		m_arenasAcc[0].m_dataAcc[0].head.data = 48;
	}

	int GetFakeAllocData() const
	{
		return m_arenasAcc[0].m_dataAcc[0].head.data;
	}
};
*/

/*
struct MemPoolHostStorage 
{
	std::vector<Arena<cl::sycl::access::target::global_buffer>> m_arenasOnDeviceVec;
	std::vector<Arena<cl::sycl::access::target::host_buffer>> m_arenasOnHostVec;
	std::vector<std::pair<size_t, size_t>> m_memSizeAllocSizes;
	std::vector<ArenaStorage> m_arenasStorage;

	MemPoolHostStorage(const std::vector<std::pair<size_t, size_t>> memSizeAllocSizes) :
		m_arenasOnDeviceVec(memSizeAllocSizes.size()),
		m_memSizeAllocSizes(memSizeAllocSizes)
	{
		m_arenasStorage.reserve(m_memSizeAllocSizes.size());

		for (size_t i = 0; i < m_memSizeAllocSizes.size(); ++i)
		{
			const auto& memSizeAllocSize = m_memSizeAllocSizes[i];
			m_arenasStorage.push_back(ArenaStorage(memSizeAllocSize.first, memSizeAllocSize.second));
		}

		for (auto& arenaStorage : m_arenasStorage)
		{
			m_arenasOnHostVec.push_back(Arena<cl::sycl::access::target::host_buffer>(arenaStorage));
		}
	}
};

struct MemPoolStorage
{
    cl::sycl::buffer<Arena<cl::sycl::access::target::global_buffer>> m_arenasOnDevice;
    cl::sycl::buffer<Arena<cl::sycl::access::target::host_buffer>> m_arenasOnHost;
	cl::sycl::buffer<ArenaStorage> m_arenasStorage;

    MemPoolStorage(MemPoolHostStorage& storage):
        m_arenasOnDevice(storage.m_arenasOnDeviceVec.data(), cl::sycl::range<1>(storage.m_memSizeAllocSizes.size())),
        m_arenasOnHost(storage.m_arenasOnHostVec.data(), cl::sycl::range<1>(storage.m_memSizeAllocSizes.size())),
		m_arenasStorage(storage.m_arenasStorage.data(), cl::sycl::range<1>(storage.m_arenasStorage.size()))
	{
    }

	void RefreshDeviceBuffer(cl::sycl::handler& cgh)
	{
		auto arenasHostAcc = m_arenasOnDevice.get_access<cl::sycl::access::mode::read_write>();
		auto arenasStorageAcc = m_arenasStorage.get_access<cl::sycl::access::mode::read_write>();
		for (size_t i = 0; i < m_arenasOnDevice.get_size(); ++i)
		{
			arenasHostAcc[i].RefreshForDevice(arenasStorageAcc[i], cgh);
		}
	}
};
*/

/*
template<cl::sycl::access::target target>
struct MemPool
{
    cl::sycl::accessor<Arena<target>, 1, cl::sycl::access::mode::read_write, target> m_arenasAcc;

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::global_buffer>>
	MemPool(MemPoolStorage& storage, cl::sycl::handler& cgh) : m_arenasAcc(storage.m_arenasOnDevice.get_access<cl::sycl::access::mode::read_write, target>(cgh))
	{
	}

	template <cl::sycl::access::target target2 = target, typename = std::enable_if_t<target2 == cl::sycl::access::target::host_buffer>>
	MemPool(MemPoolStorage& storage) : m_arenasAcc(storage.m_arenasOnHost.get_access<cl::sycl::access::mode::read_write>())
	{
	}

    void FakeAlloc() const
    {
        m_arenasAcc[0].m_dataAcc[0].head.data = 48;
    }

    int GetFakeAllocData() const
    {
        return m_arenasAcc[0].m_dataAcc[0].head.data;
    }
};*/

struct NestedData
{
	int data;
};

struct MiddleData
{
	cl::sycl::buffer<NestedData> m_nestedData;

	MiddleData() : m_nestedData(cl::sycl::range<1>(4)) {}
};

struct MiddleDataView
{
	cl::sycl::accessor<MiddleData, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> m_nestedDataAcc;
};

struct OuterData
{
	cl::sycl::buffer<MiddleData> m_middleData;
	OuterData(const size_t& size) : m_middleData(cl::sycl::range<1>(size))
	{
		auto acc = m_middleData.get_access<cl::sycl::access::mode::read_write>();
		for (size_t i = 0; i < size; ++i)
		{
			m_middleData[acc] = MiddleData();
		}
	}
};

struct OuterDataView
{
	cl::sycl::accessor<OuterData, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> m_middleDataAcc;
};

class init_first_block;

int main(int argc, char** argv)
{
	/*ArenaStorageBlock sharedStorageBlock({ { 1024, 64 },{ 2048, 128 },{ 8192, 1024 } });
	
	MemPoolVecStorage<cl::sycl::access::target::global_buffer> hostVecStorage(sharedStorageBlock);


	MemPoolHostStorage hostStorage({ { 1024, 64 },{ 2048, 128 },{ 8192, 1024 } });
    MemPoolStorage memPoolStorage(hostStorage);*/

    cl::sycl::queue workQueue;
	workQueue.submit([&](cl::sycl::handler& cgh) {
        
		memPoolStorage.RefreshDeviceBuffer(cgh);
        MemPool<cl::sycl::access::target::global_buffer> memPool(memPoolStorage, cgh);

        cgh.single_task<class init_first_block> ([memPool]() {
            memPool.FakeAlloc();
        });
    });
    
	workQueue.wait();

    MemPool<cl::sycl::access::target::host_buffer> memPoolOnHost(memPoolStorage);

    std::cout << "result of fake alloc (should be 48): " << memPoolOnHost.GetFakeAllocData() << std::endl;

    return 0;
}
