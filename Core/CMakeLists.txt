FILE(GLOB SOURCE_LIST *.cpp *.h *.hpp)
add_library(FunGPUCore ${SOURCE_LIST})

add_sycl_to_target(
        TARGET "FunGPUCore"
        SOURCES ${SOURCE_LIST})

set_property(TARGET FunGPUCore PROPERTY CXX_STANDARD 14)
