cmake_minimum_required(VERSION 3.0.0)
project(nano_mind VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(lib)
add_subdirectory(vendor)
add_subdirectory(tests)

add_executable(nano_mind main.cpp 
    src/sgd_optimizer.cpp
)
set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
target_include_directories(
    nano_mind
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(nano_mind loss_functions layers core algorithms utils)
