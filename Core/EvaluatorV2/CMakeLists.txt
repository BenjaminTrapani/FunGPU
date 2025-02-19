add_library(EvaluatorV2 BlockGenerator.cpp Instruction.cpp Lambda.cpp Program.cpp CompileProgram.cpp Evaluator.cpp)
add_sycl_to_target(
  TARGET
    EvaluatorV2
)
target_link_libraries(EvaluatorV2 PUBLIC FunGPUCore)

add_executable(FunGPUV2 FunGPUV2.cpp)
target_link_libraries(FunGPUV2 EvaluatorV2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_subdirectory(tests)
