cmake_minimum_required(VERSION 3.4.3)

set(ComputeCpp_DIR "C:/Program Files/Codeplay/ComputeCpp/")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

include_directories(SYSTEM "${ComputeCpp_INCLUDE_DIRS}")
include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")

FILE(GLOB SOURCE_LIST *.cpp *.h *.hpp)
add_executable(SyclTests ${SOURCE_LIST})

add_sycl_to_target(
    TARGET "SyclTests"
    SOURCES ${SOURCE_LIST})
