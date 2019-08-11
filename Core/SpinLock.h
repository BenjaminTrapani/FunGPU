#pragma once

#include "SYCL/sycl.hpp"
#include <atomic>

namespace FunGPU {
class SpinLock {
public:
  SpinLock() : m_countData(0) {}

  SpinLock(const SpinLock &lock) : m_countData(0) {}

  void Aquire() {
    int expectedCount = 0;
    cl::sycl::atomic<int> count((
        cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(
            &m_countData)));
    while (!count.compare_exchange_strong(expectedCount, 1)) {
      expectedCount = 0;
    }
  }

  void Release() {
    cl::sycl::atomic<int> count((
        cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space>(
            &m_countData)));
    count.store(0);
  }

private:
  int m_countData;
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
