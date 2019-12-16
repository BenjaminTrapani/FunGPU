cmake_minimum_required(VERSION 3.4.3)

project(FunGPU)
enable_testing()

add_executable(FunGPU main.cpp)

add_subdirectory(Core)
add_subdirectory(TestPrograms)
add_subdirectory(IntegrationTests)

target_link_libraries(FunGPU FunGPUCore)

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h *.hpp)
find_program (CLANGFORMAT_CMD clang-format "C:/Program Files/LLVM/bin/")

add_custom_target(clangformat
    COMMAND ${CLANGFORMAT_CMD}
    -i
    ${ALL_SOURCE_FILES})
