#pragma once

#include "Core/PortableMemPool.hpp"
#include "Core/Program.hpp"
#include "Core/RuntimeValue.h"
#include <CL/sycl.hpp>

namespace FunGPU::EvaluatorV2 {
class Evaluator {
public:
  Evaluator(cl::sycl::buffer<PortableMemPool>);
  RuntimeValue compute(const Program &);

private:
  cl::sycl::buffer<PortableMemPool> mem_pool_buffer_;
};
} // namespace FunGPU::EvaluatorV2
