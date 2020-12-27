add_library(EvaluatorV2 BlockGenerator.cpp Instruction.cpp Lambda.cpp Program.cpp CompileProgram.cpp Evaluator.cpp)
target_link_libraries(EvaluatorV2 FunGPUCore)

add_executable(FunGPUV2 FunGPUV2.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "FunGPUV2"
          SOURCES FunGPUV2.cpp)
endif()
target_link_libraries(FunGPUV2 EvaluatorV2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_subdirectory(tests)
