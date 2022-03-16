include(GoogleTest)

add_library(GooStatsTestFit fit.cc)
add_sanitizers(GooStatsTestFit)
target_link_libraries(GooStatsTestFit PUBLIC GooStatsLib)
target_include_directories(GooStatsTestFit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(COPY toyMC.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY NLL_CHECK_reference.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_GooStats simpleFit.cc)
add_sanitizers(test_GooStats)
target_link_libraries(test_GooStats PRIVATE GooStatsTestFit gtest_main)
gtest_discover_tests(test_GooStats)

add_library(GooStatsTestFit_NLL fit.cc)
add_sanitizers(GooStatsTestFit_NLL)
target_link_libraries(GooStatsTestFit_NLL PUBLIC GooStatsLib_NLL)
target_include_directories(GooStatsTestFit_NLL
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_GooStatsNLL NLLtest.cc)
add_sanitizers(test_GooStatsNLL)
target_link_libraries(
  test_GooStatsNLL PRIVATE GooStatsTestFit_NLL GooStatsLib_NLL gtest_main
                           ROOT::RIO)

add_test(NAME test_GooStatsNLL COMMAND $<TARGET_FILE:test_GooStatsNLL>
                                       --gtest_filter=GooStats.NLLTest)
add_test(NAME test_GooStatsNLLFail COMMAND $<TARGET_FILE:test_GooStatsNLL>
                                           --gtest_filter=GooStats.NLLTestFail)

set_tests_properties(test_GooStatsNLLFail PROPERTIES WILL_FAIL TRUE)

add_subdirectory(Kernel)
add_subdirectory(SimplePackage)
