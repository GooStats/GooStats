file(COPY ../data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(SimpleInputBuilderTest SimpleInputBuilderTest.cc)
add_sanitizers(SimpleInputBuilderTest)
target_link_libraries(SimpleInputBuilderTest PRIVATE GooStatsLib gtest_main)
add_test(NAME SimpleInputBuilderTest COMMAND SimpleInputBuilderTest)
