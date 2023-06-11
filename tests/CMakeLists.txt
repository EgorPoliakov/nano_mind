if (BUILD_TESTING)
    add_executable(tests 
        src/derivatives.cpp 
        src/activations.cpp
        src/arithmetic_functions.cpp
        src/neuron.cpp
        src/losses.cpp
    )
    target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(tests GTest::gtest_main core algorithms layers loss_functions)
    include(GoogleTest)
    gtest_discover_tests(tests)
endif()