#include "Core/Evaluator.hpp"

namespace FunGPU::EvaluatorV2 {
Evaluator::Evaluator(cl::sycl::buffer<PortableMemPool> buffer)
    : mem_pool_buffer_(buffer) {}

RuntimeValue Evaluator::compute(const Program &program) {}
} // namespace FunGPU::EvaluatorV2
