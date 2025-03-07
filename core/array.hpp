#include "core/portable_mem_pool.hpp"
#include "core/types.hpp"

namespace FunGPU {
template <class T> class Array {
public:
  Array(const Index_t size, PortableMemPool *pool)
      : m_size(size), m_pool(pool) {
    if (m_size != 0) {
      m_data = m_pool->alloc_array<T>(m_size);
    }
  }

  ~Array() {
    if (m_size != 0) {
      m_pool->dealloc_array(m_data);
    }
  }

  T &get(const Index_t index) {
    auto data_ref = m_pool->deref_handle(m_data);
    return data_ref[index];
  }

  void set(const Index_t index, const T &val) {
    auto data_ref = m_pool->deref_handle(m_data);
    data_ref[index] = val;
  }

  Index_t size() const { return m_size; }

private:
  PortableMemPool::ArrayHandle<T> m_data;
  const Index_t m_size;
  PortableMemPool *m_pool;
};
} // namespace FunGPU
