if (BUILD_TESTING)
    add_executable(tests 
        src/derivatives.cpp 
        src/activations.cpp
        src/neuron.cpp
    )
    target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(tests GTest::gtest_main core algorithms)
    include(GoogleTest)
    gtest_discover_tests(tests)
endif()