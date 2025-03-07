add_library(evaluator_v2 block_generator.cpp instruction.cpp lambda.cpp program.cpp compile_program.cpp evaluator.cpp)
add_sycl_to_target(
  TARGET
    evaluator_v2
)
target_link_libraries(evaluator_v2 PUBLIC fungpu_core)

add_executable(fungpu_v2 fun_gpu_v2.cpp)
target_link_libraries(fungpu_v2 evaluator_v2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_subdirectory(tests)
