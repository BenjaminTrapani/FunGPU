#pragma once

#include "Core/EvaluatorV2/Program.hpp"
#include "Core/PortableMemPool.hpp"
#include <string>

namespace FunGPU::EvaluatorV2 {
Program compile_program(const std::string &path, Index_t registers_per_thread,
                        Index_t threads_per_block,
                        cl::sycl::buffer<PortableMemPool> &);
}
