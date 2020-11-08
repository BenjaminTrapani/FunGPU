FILE(GLOB SOURCE_LIST)

add_library(FunGPUCore Compiler.cpp Parser.cpp SExpr.cpp BlockPrep.cpp CollectAllASTNodes.cpp)
add_library(FunGPUEvaluator CPUEvaluator.cpp)
target_link_libraries(FunGPUEvaluator FunGPUCore)

add_subdirectory(EvaluatorV2)
