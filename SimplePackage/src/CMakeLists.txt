add_library(GooStatsSimple
  SimpleDatasetController.cc
  SimpleInputBuilder.cc
  SimpleOutputBuilder.cc
  SimpleSpectrumBuilder.cc
  SimplePlotManager.cc
  PrepareData.cc
  PullDatasetController.cc)
target_include_directories(GooStatsSimple PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/SimplePackage/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsSimple PUBLIC GooStatsStatModules ROOT::Gpad)
