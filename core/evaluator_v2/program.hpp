#pragma once

#include "core/evaluator_v2/lambda.hpp"
#include "core/portable_mem_pool.hpp"

namespace FunGPU::EvaluatorV2 {
using Program = PortableMemPool::ArrayHandle<Lambda>;

std::string print(const Program &, PortableMemPool::HostAccessor_t);
void deallocate_program(const Program &, PortableMemPool::HostAccessor_t);
} // namespace FunGPU::EvaluatorV2
