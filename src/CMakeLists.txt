add_library(GooStatsExperimental
  DeclareModule.cc
  ModuleFactory.cc
  ModuleManager.cc)
target_include_directories(GooStatsExperimental PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsExperimental PUBLIC GooStatsKernel)
