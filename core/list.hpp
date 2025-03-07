#include "core/error.hpp"
#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"
#include <CL/sycl.hpp>
#include <memory>

namespace FunGPU {
template <class T> class List {
public:
  List(const PortableMemPool::DeviceAccessor_t &memPool)
      : m_portableMemPool(memPool) {}

  ~List() {
    while (m_head != PortableMemPool::Handle<ListNode>()) {
      auto derefdHead = m_portableMemPool[0].deref_handle(m_head);
      auto nextRef = derefdHead->m_next;
      m_portableMemPool[0].dealloc(m_head);
      m_head = nextRef;
    }
  }

  void set_mem_pool_acc(const PortableMemPool::DeviceAccessor_t &acc) {
    m_portableMemPool = acc;
  }

  Index_t size() const { return m_listSize; }
  Error push_front(const T &val) {
    const auto newNodeHandle =
        m_portableMemPool[0].template alloc<ListNode>(val);
    if (newNodeHandle == PortableMemPool::Handle<ListNode>()) {
      return Error(Error::Type::MemPoolAllocFailure);
    }

    auto newNode = m_portableMemPool[0].deref_handle(newNodeHandle);

    newNode->m_next = m_head;
    m_head = newNodeHandle;
    ++m_listSize;

    return Error();
  }

  PortableMemPool::Handle<T> front() { return m_head; }

  T &deref_front() { return *m_portableMemPool[0].deref_handle(front()); }

  template <class Callable_t> void map(Callable_t &callable) {
    auto tempHead = m_head;
    while (tempHead != PortableMemPool::Handle<ListNode>()) {
      auto tempHeadDerefd = m_portableMemPool[0].deref_handle(tempHead);
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

    auto derefdHead = m_portableMemPool[0].deref_handle(m_head);
    auto nextHead = derefdHead->m_next;
    m_portableMemPool[0].dealloc(m_head);
    m_head = nextHead;
    --m_listSize;
  }

  PortableMemPool::Handle<T> get_item_at_index(Index_t index) {
    auto tempHead = m_head;
    while (index > 0 && tempHead != PortableMemPool::Handle<ListNode>()) {
      auto tempHeadDerefd = m_portableMemPool[0].deref_handle(tempHead);
      tempHead = tempHeadDerefd->m_next;
      --index;
    }
    if (tempHead == PortableMemPool::Handle<ListNode>()) {
      return PortableMemPool::Handle<T>();
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
