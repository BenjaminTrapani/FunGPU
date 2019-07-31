cmake_minimum_required(VERSION 3.4.3)

project(FunGPU)

enable_testing()

set(ComputeCpp_DIR "/home/benjamintrapani/libs/ComputeCpp-CE-1.1.4-Ubuntu-16.04-x86_64")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

include_directories(SYSTEM "${ComputeCpp_INCLUDE_DIRS}")
include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")

FILE(GLOB SOURCE_LIST *.cpp *.h *.hpp)
add_executable(FunGPU ${SOURCE_LIST})

add_subdirectory(Core)
add_subdirectory(TestPrograms)
add_subdirectory(IntegrationTests)

target_link_libraries(FunGPU FunGPUCore)

add_sycl_to_target(
    TARGET "FunGPU"
    SOURCES ${SOURCE_LIST})

set_property(TARGET FunGPU PROPERTY CXX_STANDARD 14)

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h *.hpp)
find_program (CLANGFORMAT_CMD clang-format)

add_custom_target(clangformat
    COMMAND ${CLANGFORMAT_CMD}
    -i
    ${ALL_SOURCE_FILES})
