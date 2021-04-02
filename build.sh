#!/usr/bin/env bash
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=syclcc-clang -DCMAKE_BUILD_TYPE=Release -G Ninja ../
ninja all
