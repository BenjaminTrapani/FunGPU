cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

project(FunGPU)
enable_testing()

add_executable(FunGPU main.cpp)

add_subdirectory(Core)
add_subdirectory(TestPrograms)
add_subdirectory(IntegrationTests)

target_link_libraries(FunGPU FunGPUCore)

file(GLOB_RECURSE ALL_SOURCE_FILES Core/*.cpp Core/*.h Core/*.hpp main.cpp)
find_program (CLANGFORMAT_CMD clang-format "/usr/lib/llvm-9/bin/")

add_custom_target(clangformat
    COMMAND ${CLANGFORMAT_CMD}
    -i
    ${ALL_SOURCE_FILES})
