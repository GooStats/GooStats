add_library(GooStatsSimple
  SimpleDatasetController.cc
  SimpleInputBuilder.cc
  SimpleOutputBuilder.cc
  SimpleSpectrumBuilder.cc
  SimplePlotManager.cc
  PrepareData.cc
  PullDatasetController.cc)
target_link_libraries(GooStatsSimple GooStatsKernel GooStatsPDFs)
