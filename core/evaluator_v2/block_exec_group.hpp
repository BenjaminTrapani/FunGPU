#pragma once

#include "core/types.hpp"

namespace FunGPU {
struct BlockExecGroup {
    Index_t num_blocks = 0;
    Index_t max_num_instructions = 0;
  };
}
