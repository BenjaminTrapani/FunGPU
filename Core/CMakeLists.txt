FILE(GLOB SOURCE_LIST)
add_library(FunGPUCore Compiler.cpp CPUEvaluator.cpp Parser.cpp SExpr.cpp)
