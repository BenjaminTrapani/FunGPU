#include "Types.h"
#include "PortableMemPool.h"
#include <memory>
#include "SYCL/sycl.hpp"

namespace FunGPU
{
	template<class T>
	class List
	{
	public:
		List(const PortableMemPool::DeviceAccessor_t& memPool): m_portableMemPool(memPool)
		{
		}

		~List()
		{
			while (m_head != PortableMemPool::Handle<ListNode>())
			{
				auto derefdHead = m_portableMemPool->derefHandle(m_head);
				auto nextRef = derefdHead->m_next;
				m_portableMemPool->Dealloc(m_head);
				m_head = nextRef;
			}
		}

		Index_t size() const
		{
			return m_listSize;
		}
		void push_front(const T& val)
		{
			const auto newNodeHandle = m_portableMemPool->Alloc<ListNode>(val);
			auto newNode = m_portableMemPool->derefHandle(newNodeHandle);

			newNode->m_next = m_head;
			m_head = newNodeHandle;
			++m_listSize;
		}

		PortableMemPool::Handle<T> front()
		{
			return m_head;
		}

		T& derefFront()
		{
			return *m_portableMemPool->derefHandle(front());
		}

		void pop_front()
		{
			if (m_head == PortableMemPool::Handle<ListNode>())
			{
				throw std::invalid_argument("Cannot delete head from list when it doesn't exist");
			}

			auto derefdHead = m_portableMemPool->derefHandle(m_head);
			auto nextHead = derefdHead->m_next;
			m_portableMemPool->Dealloc(m_head);
			m_head = nextHead;
			--m_listSize;
		}

		PortableMemPool::Handle<T> GetItemAtIndex(Index_t index)
		{
			auto tempHead = m_head;
			while (index > 0 && tempHead != PortableMemPool::Handle<ListNode>())
			{
				auto tempHeadDerefd = m_portableMemPool->derefHandle(tempHead);
				tempHead = tempHeadDerefd->m_next;
				--index;
			}
			if (tempHead == PortableMemPool::Handle<ListNode>())
			{
				throw std::invalid_argument("Index out of range in list");
			}
			return tempHead;
		}

	private:

		class ListNode : public T
		{
		public:
			using ListNodeHandle_t = PortableMemPool::Handle<ListNode>;
			ListNode(const T& val) : T(val) {}

			ListNodeHandle_t m_next;
		};

		typename ListNode::ListNodeHandle_t m_head;
		Index_t m_listSize = 0;
		PortableMemPool::DeviceAccessor_t m_portableMemPool;
	};
}
