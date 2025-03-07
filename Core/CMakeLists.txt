FILE(GLOB SOURCE_LIST)

add_library(fungpu_core compiler.cpp parser.cpp s_expr.cpp block_prep.cpp collect_all_ast_nodes.cpp)
add_sycl_to_target(TARGET fungpu_core)

add_library(fungpu_evaluator cpu_evaluator.cpp)
target_link_libraries(fungpu_evaluator fungpu_core)

add_subdirectory(EvaluatorV2)
