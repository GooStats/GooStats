add_library(GooStatsTestFit fit.cc)
target_link_libraries(GooStatsTestFit PUBLIC GooStatsLib)
target_include_directories(GooStatsTestFit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(COPY toyMC.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY NLL_CHECK_reference.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_GooStats simpleFit.cc)
target_link_libraries(test_GooStats PRIVATE GooStatsTestFit gtest_main)
add_test(NAME test_GooStats
        COMMAND $<TARGET_FILE:test_GooStats>)

add_library(GooStatsTestFit_NLL fit.cc)
target_link_libraries(GooStatsTestFit_NLL PUBLIC GooStatsLib_NLL)
target_include_directories(GooStatsTestFit_NLL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(runGooStatsTestFit_NLL run.cc)
target_link_libraries(runGooStatsTestFit_NLL PRIVATE GooStatsTestFit_NLL)

add_executable(test_GooStatsNLL NLLtest.cc)
target_link_libraries(test_GooStatsNLL PRIVATE GooStatsLib_NLL gtest_main ROOT::RIO)

add_test(NAME test_GooStatsNLL
        COMMAND bash -c "$<TARGET_FILE:runGooStatsTestFit_NLL>;
        $<TARGET_FILE:test_GooStatsNLL> --gtest_filter=GooFit.NLLTest")

add_test(NAME test_GooStatsNLLFail
        COMMAND bash -c
        "$<TARGET_FILE:runGooStatsTestFit_NLL>;
        $<TARGET_FILE:test_GooStatsNLL> --gtest_filter=GooFit.NLLTestFail")
set_tests_properties(test_GooStatsNLLFail PROPERTIES WILL_FAIL TRUE)