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
