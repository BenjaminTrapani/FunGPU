#!/usr/bin/env bash
mkdir build
cd build
export HIPSYCL_TARGETS=cuda:sm_61
export HIPSYCL_CUDA_PATH=/usr/local/cuda-11.2/
cmake -DCMAKE_CXX_COMPILER=syclcc-clang -DCMAKE_BUILD_TYPE=Release -G Ninja ../
ninja all
