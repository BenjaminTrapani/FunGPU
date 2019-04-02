#include "Types.h"

namespace FunGPU
{
	template<class T>
	class List
	{
	public:
		Index_t size() const
		{
			return m_listSize;
		}
		void push_front(const T& val)
		{
			ListNode* newNode = new ListNode(val);
			newNode->m_next = m_head;
			m_head = newNode;
			++m_listSize;
		}
		T& front()
		{
			return m_head->m_data;
		}

		void pop_front()
		{
			if (m_head == nullptr)
			{
				throw std::invalid_argument("Cannot delete head from list when it doesn't exist");
			}
			auto nextHead = m_head->m_next;
			delete m_head;
			m_head = nextHead;
			--m_listSize;
		}
		T& GetItemAtIndex(Index_t index)
		{
			auto tempHead = m_head;
			while (index > 0 && tempHead)
			{
				tempHead = tempHead->m_next;
				--index;
			}
			if (tempHead == nullptr)
			{
				throw std::invalid_argument("Index out of range in list");
			}
			return tempHead->m_data;
		}
	private:
		class ListNode
		{
		public:
			ListNode(const T& val): m_data(val) {}

			T m_data;
			ListNode* m_next = nullptr;
			ListNode* m_prev = nullptr;
		};

		ListNode* m_head = nullptr;
		Index_t m_listSize = 0;
	};
}