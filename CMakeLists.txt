cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -g")

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

add_executable(fungpu main.cpp)

add_subdirectory(core)
add_subdirectory(test_programs)
add_subdirectory(integration_tests)

target_link_libraries(fungpu fungpu_evaluator)
