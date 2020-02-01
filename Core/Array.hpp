#include "PortableMemPool.hpp"
#include "Types.hpp"

namespace FunGPU {
template <class T> class Array {
public:
  Array(const Index_t size, PortableMemPool *pool)
      : m_size(size), m_pool(pool) {
    if (m_size != 0) {
      m_data = m_pool->AllocArray<T>(m_size);
    }
  }

  ~Array() {
    if (m_size != 0) {
      m_pool->DeallocArray(m_data);
    }
  }

  T &Get(const Index_t index) {
    auto dataRef = m_pool->derefHandle(m_data);
    return dataRef[index];
  }
  void Set(const Index_t index, const T &val) {
    auto dataRef = m_pool->derefHandle(m_data);
    dataRef[index] = val;
  }
  Index_t size() const { return m_size; }

private:
  PortableMemPool::ArrayHandle<T> m_data;
  const Index_t m_size;
  PortableMemPool *m_pool;
};
} // namespace FunGPU
