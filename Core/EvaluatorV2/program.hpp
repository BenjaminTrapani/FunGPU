#pragma once

#include "Core/EvaluatorV2/lambda.hpp"
#include "Core/portable_mem_pool.hpp"

namespace FunGPU::EvaluatorV2 {
using Program = PortableMemPool::ArrayHandle<Lambda>;

std::string print(const Program &, PortableMemPool::HostAccessor_t);
void deallocate_program(const Program &, PortableMemPool::HostAccessor_t);
} // namespace FunGPU::EvaluatorV2
