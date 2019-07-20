#pragma once
#include "SYCL/sycl.hpp"
namespace FunGPU
{
	template<class Buffer_t>
	class Sycl1DBufferWithAccessors {
	public:
		Sycl1DBufferWithAccessors(const Buffer_t& buffer): m_buffer(buffer)
		{
		}

		void ConfigureForDevice(cl::sycl::handler& h)
		{
			m_deviceAcc = m_buffer.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(h);
		}
		
		void ConfigureForHost(cl::sycl::handler& h)
		{
			m_hostAcc = m_buffer.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(h);
		}
		
		cl::sycl::accessor<typename Buffer_t::value_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> GetHostAcc()
		{
			return m_hostAcc
		}

		cl::sycl::accessor<typename Buffer_t::value_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> GetDeviceAcc()
		{
			return m_deviceAcc;
		}

	private:
		Buffer_t m_buffer;
		cl::sycl::accessor<typename Buffer_t::value_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> m_hostAcc;
		cl::sycl::accessor<typename Buffer_t::value_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> m_deviceAcc;
	};
}