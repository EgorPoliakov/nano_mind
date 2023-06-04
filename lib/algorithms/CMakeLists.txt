add_library(algorithms 
    src/topological_sort.cpp
)

target_include_directories(algorithms PRIVATE 
    include
    ${CMAKE_SOURCE_DIR}/lib/core/include
)