add_library(loss_functions 
    src/mse_loss.cpp
    src/cross_entropy_loss.cpp
)

target_include_directories(loss_functions PUBLIC 
    include 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib/core/include
    ${CMAKE_SOURCE_DIR}/lib/layers/include
)