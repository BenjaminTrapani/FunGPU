#pragma once

#include "Core/EvaluatorV2/Lambda.hpp"
#include "Core/PortableMemPool.hpp"

namespace FunGPU::EvaluatorV2 {
  using Program = PortableMemPool::ArrayHandle<Lambda>;
}
