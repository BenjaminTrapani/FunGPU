FILE(GLOB SOURCE_LIST)

add_library(FunGPUCore Compiler.cpp Parser.cpp SExpr.cpp BlockPrep.cpp CollectAllASTNodes.cpp CPUEvaluator.cpp)

add_subdirectory(EvaluatorV2)
