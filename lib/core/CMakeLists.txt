add_library(core 
    src/model.cpp
    src/neuron.cpp
    src/parameter.cpp
)

target_include_directories(core PUBLIC 
    include 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib/layers/include
    ${CMAKE_SOURCE_DIR}/lib/algorithms/include
)