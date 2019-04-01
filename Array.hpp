#include "Types.h"

namespace FunGPU
{
	template <class T>
	class Array
	{
	public:
		Array(const Index_t size) : m_data(new T[size]), m_size(size) {}
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