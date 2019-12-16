#include "PortableMemPool.hpp"
#include <CL/sycl.hpp>
#include "Types.h"
#include <memory>

namespace FunGPU {
template <class T> class List {
public:
  List(const PortableMemPool::DeviceAccessor_t &memPool)
      : m_portableMemPool(memPool) {}

  ~List() {
    while (m_head != PortableMemPool::Handle<ListNode>()) {
      auto derefdHead = m_portableMemPool[0].derefHandle(m_head);
      auto nextRef = derefdHead->m_next;
      m_portableMemPool[0].Dealloc(m_head);
      m_head = nextRef;
    }
  }

  void SetMemPoolAcc(const PortableMemPool::DeviceAccessor_t &acc) {
    m_portableMemPool = acc;
  }

  Index_t size() const { return m_listSize; }
  bool push_front(const T &val) {
    const auto newNodeHandle =
        m_portableMemPool[0].template Alloc<ListNode>(val);
    if (newNodeHandle == PortableMemPool::Handle<ListNode>()) {
      return false;
    }

    auto newNode = m_portableMemPool[0].derefHandle(newNodeHandle);

    newNode->m_next = m_head;
    m_head = newNodeHandle;
    ++m_listSize;

    return true;
  }

  PortableMemPool::Handle<T> front() { return m_head; }

  T &derefFront() { return *m_portableMemPool[0].derefHandle(front()); }

  template <class Callable_t> void map(Callable_t &callable) {
    auto tempHead = m_head;
    while (tempHead != PortableMemPool::Handle<ListNode>()) {
      auto tempHeadDerefd = m_portableMemPool[0].derefHandle(tempHead);
      callable(*static_cast<T *>(tempHeadDerefd));
      tempHead = tempHeadDerefd->m_next;
    }
  }

  void pop_front() {
    if (m_head == PortableMemPool::Handle<ListNode>()) {
      return;
      // throw std::invalid_argument("Cannot delete head from list when it
      // doesn't exist");
    }

    auto derefdHead = m_portableMemPool[0].derefHandle(m_head);
    auto nextHead = derefdHead->m_next;
    m_portableMemPool[0].Dealloc(m_head);
    m_head = nextHead;
    --m_listSize;
  }

  PortableMemPool::Handle<T> GetItemAtIndex(Index_t index) {
    auto tempHead = m_head;
    while (index > 0 && tempHead != PortableMemPool::Handle<ListNode>()) {
      auto tempHeadDerefd = m_portableMemPool[0].derefHandle(tempHead);
      tempHead = tempHeadDerefd->m_next;
      --index;
    }
    if (tempHead == PortableMemPool::Handle<ListNode>()) {
      return PortableMemPool::Handle<T>();
      // throw std::invalid_argument("Index out of range in list");
    }
    return tempHead;
  }

private:
  class ListNode : public T {
  public:
    using ListNodeHandle_t = PortableMemPool::Handle<ListNode>;
    ListNode(const T &val) : T(val) {}

    ListNodeHandle_t m_next;
  };

  typename ListNode::ListNodeHandle_t m_head;
  Index_t m_listSize = 0;
  PortableMemPool::DeviceAccessor_t m_portableMemPool;
};
} // namespace FunGPU
