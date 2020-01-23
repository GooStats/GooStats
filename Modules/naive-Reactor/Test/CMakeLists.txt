################
# Define a test

add_executable(autoTest autoTest.cc)
target_link_libraries(autoTest PRIVATE TestFixture ${GTEST_BOTH_LIBRARIES} pthread)
target_link_libraries(autoTest PRIVATE reactor_module)
target_link_libraries(autoTest PRIVATE GooStatsLib)
if(COVERAGE OR DEFINED ENV{COVERAGE})
  message(STATUS "Enabling COVERAGE TEST")
  target_compile_options(autoTest PRIVATE --coverage)
  target_link_libraries(autoTest PRIVATE --coverage)
endif()
add_test(NAME    autoTest
  COMMAND autoTest
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
