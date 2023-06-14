add_library(utils src/utils.cpp)
target_include_directories(utils PUBLIC 
    include 
    ${CMAKE_SOURCE_DIR}/lib/core/include
)