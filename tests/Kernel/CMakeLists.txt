add_executable(BasicManagerTest BasicManagerTest.cc)
add_sanitizers(BasicManagerTest)
target_link_libraries(BasicManagerTest PRIVATE GooStatsLib gtest_main)
add_test(NAME BasicManagerTest COMMAND BasicManagerTest)

add_executable(UtilityTest UtilityTest.cc)
add_sanitizers(UtilityTest)
target_link_libraries(UtilityTest PRIVATE GooStatsLib gtest_main)
add_test(NAME UtilityTest COMMAND UtilityTest)

add_executable(DatabaseTest DatabaseTest.cc)
add_sanitizers(DatabaseTest)
target_link_libraries(DatabaseTest PRIVATE GooStatsLib gtest_main)
add_test(NAME DatabaseTest COMMAND DatabaseTest)
