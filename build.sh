#!/usr/bin/env bash
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=syclcc-clang -G Ninja ../
ninja all
