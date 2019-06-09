#pragma once

#include <atomic>

namespace FunGPU
{
	class SpinLock
	{
	public:
		SpinLock();
		SpinLock(const SpinLock& lock);

		void Aquire();
		void Release();
	private:
		std::atomic<int> m_count;
	};

	class SpinLockGuard
	{
	public:
		SpinLockGuard(SpinLock& lock);
		~SpinLockGuard();
	private:
		SpinLock& m_lock;
	};
}