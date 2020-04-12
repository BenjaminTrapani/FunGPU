add_executable(IntegrationTestsExe IntegrationTests.cpp)

find_package(Boost 1.58 REQUIRED COMPONENTS unit_test_framework)

target_include_directories(IntegrationTestsExe PRIVATE ${CMAKE_SOURCE_DIR}/Core ${Boost_INCLUDE_DIR})

add_test(NAME IntegrationTests COMMAND IntegrationTestsExe)

target_compile_definitions(IntegrationTestsExe PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(IntegrationTestsExe FunGPUCore ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
