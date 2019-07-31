#pragma once

#include <atomic>
#include "SYCL/sycl.hpp"

namespace FunGPU
{
	class SpinLock
	{
	public:
        SpinLock(): m_countData(0),
                    m_count(cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(&m_countData))
        {
        }

        SpinLock(const SpinLock& lock) : m_countData(0),
                                         m_count(cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(&m_countData))
        {
        }

        void Aquire()
        {
            int expectedCount = 0;
            while (!m_count.compare_exchange_strong(expectedCount, 1))
            {
                expectedCount = 0;
            }
        }

        void Release()
        {
            m_count.store(0);
        }
	private:
	    int m_countData;
        cl::sycl::atomic<int> m_count;
	};

	/*class SpinLockGuard
	{
	public:
        SpinLockGuard(SpinLock& lock): m_lock(lock)
        {
            m_lock.Aquire();
        }
        ~SpinLockGuard()
        {
            m_lock.Release();
        }

	private:
		SpinLock& m_lock;
	};*/
}
