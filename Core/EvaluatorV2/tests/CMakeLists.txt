add_executable(BlockGeneratorTestExe BlockGenerator.cpp)
target_compile_definitions(BlockGeneratorTestExe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(BlockGeneratorTestExe EvaluatorV2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME BlockGeneratorTest WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND BlockGeneratorTestExe)

add_executable(RuntimeBlockTestExe RuntimeBlock.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "RuntimeBlockTestExe"
          SOURCES RuntimeBlock.cpp)
endif()
target_compile_definitions(RuntimeBlockTestExe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(RuntimeBlockTestExe EvaluatorV2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME RuntimeBlockTest WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND RuntimeBlockTestExe)

add_executable(IndirectCallHandlerTestExe IndirectCallHandler.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "IndirectCallHandlerTestExe"
          SOURCES IndirectCallHandler.cpp)
endif()
target_compile_definitions(IndirectCallHandlerTestExe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(IndirectCallHandlerTestExe EvaluatorV2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME IndirectCallHandlerTest WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND IndirectCallHandlerTestExe)

add_executable(EvaluatorTestExe Evaluator.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "EvaluatorTestExe"
          SOURCES Evaluator.cpp)
endif()
target_compile_definitions(EvaluatorTestExe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(EvaluatorTestExe EvaluatorV2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME EvaluatorTest WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND EvaluatorTestExe)
