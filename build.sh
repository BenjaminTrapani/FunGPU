#!/usr/bin/env bash
mkdir build
cd build
export HIPSYCL_GPU_ARCH=sm_61
cmake -DCMAKE_CXX_COMPILER=syclcc-clang -DCMAKE_BUILD_TYPE=Release -G Ninja ../
ninja all
