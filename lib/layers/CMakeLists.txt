add_library(layers 
    src/layer.cpp
    src/linear_layer.cpp
    src/relu_layer.cpp
)

target_include_directories(layers PUBLIC include ${CMAKE_SOURCE_DIR}/include)