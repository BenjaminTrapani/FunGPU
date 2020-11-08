add_library(EvaluatorV2 BlockGenerator.cpp)
target_link_libraries(EvaluatorV2 FunGPUCore)

add_subdirectory(tests)
