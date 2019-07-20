cmake_minimum_required(VERSION 3.4.3)

project(FunGPU)

set(ComputeCpp_DIR "C:/Program Files/Codeplay/ComputeCpp/")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

include_directories(SYSTEM "${ComputeCpp_INCLUDE_DIRS}")
include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")

FILE(GLOB SOURCE_LIST *.cpp *.h *.hpp)
add_executable(FunGPU ${SOURCE_LIST})

add_sycl_to_target(
    TARGET "FunGPU"
    SOURCES ${SOURCE_LIST})

set_property(TARGET FunGPU PROPERTY CXX_STANDARD 17)

add_subdirectory(TestSycl)

configure_file(TestProgram.fgpu TestProgram.fgpu COPYONLY)
configure_file(FastGetNPrimeSum.fgpu FastGetNPrimeSum.fgpu COPYONLY)
configure_file(ListExample.fgpu ListExample.fgpu COPYONLY)
configure_file(MergeSort.fgpu MergeSort.fgpu COPYONLY)
configure_file(TestInlineCalls.fgpu TestInlineCalls.fgpu COPYONLY)
configure_file(TestMultiList.fgpu TestMultiList.fgpu COPYONLY)
configure_file(TestMultiLetRec.fgpu TestMultiLetRec.fgpu COPYONLY)
