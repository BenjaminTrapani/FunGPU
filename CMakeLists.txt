cmake_minimum_required(VERSION 3.3)

project(FunGPU)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_NVCC_FLAGS -lineinfo)

FILE(GLOB SOURCES *.cpp *.h *.hpp *.cu)
cuda_add_executable(FunGPU ${SOURCES})
set_property(TARGET FunGPU PROPERTY CXX_STANDARD 17)

configure_file(TestProgram.fgpu TestProgram.fgpu COPYONLY)
