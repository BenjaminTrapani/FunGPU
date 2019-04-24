#include "Types.h"

namespace FunGPU
{
	template <class T>
	class Array
	{
	public:
		Array(const Index_t size) : m_size(size) 
		{
			if (m_size == 0)
			{
				m_data = nullptr;
			}
			else
			{
				m_data = new T[m_size];
			}
		}
		T& Get(const Index_t index)
		{
			return m_data[index];
		}
		void Set(const Index_t index, const T& val)
		{
			m_data[index] = val;
		}
		Index_t size() const
		{
			return m_size;
		}
	private:
		T* m_data;
		const Index_t m_size;
	};
}