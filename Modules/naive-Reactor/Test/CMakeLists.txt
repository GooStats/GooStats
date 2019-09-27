################
# Define a test

add_executable(autoTest autoTest.cc)
target_link_libraries(autoTest TestFixture ${GTEST_BOTH_LIBRARIES} pthread)
target_link_libraries(autoTest reactor_module)
target_link_libraries(autoTest GooStatsLib)
add_test(NAME    autoTest
           COMMAND autoTest)
