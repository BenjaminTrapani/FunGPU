add_executable(block_generator_tests_exe block_generator.cpp)
target_compile_definitions(block_generator_tests_exe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(block_generator_tests_exe evaluator_v2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME block_generator_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND block_generator_tests_exe)

add_executable(runtime_block_tests_exe runtime_block.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "runtime_block_tests_exe"
          SOURCES runtime_block.cpp)
endif()
target_compile_definitions(runtime_block_tests_exe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(runtime_block_tests_exe evaluator_v2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME runtime_block_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND runtime_block_tests_exe)

add_executable(indirect_call_handler_test_exe indirect_call_handler.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "indirect_call_handler_test_exe"
          SOURCES indirect_call_handler.cpp)
endif()
target_compile_definitions(indirect_call_handler_test_exe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(indirect_call_handler_test_exe evaluator_v2 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME indirect_call_handler_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND indirect_call_handler_test_exe)

add_executable(evaluator_test_exe evaluator.cpp)
if (ENABLE_COMPUTE_CPP)
  add_sycl_to_target(
          TARGET "evaluator_test_exe"
          SOURCES evaluator.cpp)
endif()
target_compile_definitions(evaluator_test_exe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(evaluator_test_exe evaluator_v2 fungpu_file_utils ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME evaluator_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND evaluator_test_exe)
