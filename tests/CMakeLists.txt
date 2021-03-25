add_executable(testFit fit.cc)
target_link_libraries(testFit GooStatsLib GooFit::goofit_lib gtest_main)

add_test(NAME testFit
  COMMAND $<TARGET_FILE:testFit> toyMC.cfg
  WORKING_DIRECTORY
  ${CMAKE_CURRENT_LIST_DIR})
