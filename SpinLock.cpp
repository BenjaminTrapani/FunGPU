#include "SpinLock.h"

namespace FunGPU
{
	SpinLock::SpinLock(): m_count(0)
	{
	}

	SpinLock::SpinLock(const SpinLock& lock) : m_count(0)
	{
	}

	void SpinLock::Aquire()
	{
		int expectedCount = 0;
		while (!m_count.compare_exchange_strong(expectedCount, 1))
		{
			expectedCount = 0;
		}
	}

	void SpinLock::Release()
	{
		m_count = 0;
	}

	SpinLockGuard::SpinLockGuard(SpinLock& lock): m_lock(lock)
	{
		m_lock.Aquire();
	}
	SpinLockGuard::~SpinLockGuard()
	{
		m_lock.Release();
	}
}