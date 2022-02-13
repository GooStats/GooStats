add_executable(BasicManagerTest BasicManagerTest.cc)
target_link_libraries(BasicManagerTest PRIVATE GooStatsLib gtest_main)
add_test(NAME BasicManagerTest COMMAND BasicManagerTest)

add_executable(UtilityTest UtilityTest.cc)
target_link_libraries(UtilityTest PRIVATE GooStatsLib gtest_main)
add_test(NAME UtilityTest COMMAND UtilityTest)