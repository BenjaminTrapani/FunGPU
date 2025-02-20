cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")

include_directories(${CMAKE_SOURCE_DIR})

project(FunGPU)
enable_testing()

set(ENABLE_COMPUTE_CPP OFF)

if(ENABLE_COMPUTE_CPP)
	list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
	include(FindComputeCpp)
	include_directories(SYSTEM "${ComputeCpp_INCLUDE_DIRS}")
	include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")
else()
	find_package(AdaptiveCpp REQUIRED)
	# To ensure that the system include is appended to the compile_commands.json
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/local/include/AdaptiveCpp/")
endif()

find_package(Boost 1.58 REQUIRED COMPONENTS unit_test_framework)

add_executable(FunGPU main.cpp)

add_subdirectory(Core)
add_subdirectory(TestPrograms)
add_subdirectory(IntegrationTests)

target_link_libraries(FunGPU FunGPUEvaluator)

file(GLOB_RECURSE ALL_SOURCE_FILES Core/*.cpp Core/*.h Core/*.hpp main.cpp)
find_program (CLANGFORMAT_CMD clang-format "/usr/lib/llvm-18/bin/")

add_custom_target(clangformat
    COMMAND ${CLANGFORMAT_CMD}
    -i
    ${ALL_SOURCE_FILES})
