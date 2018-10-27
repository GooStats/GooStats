add_library(GooStatsSimple
  SimpleDatasetController.cc
  SimpleInputBuilder.cc
  SimpleOptionParser.cc
  SimpleOutputBuilder.cc
  SimpleSpectrumBuilder.cc
  SimplePlotManager.cc
  PrepareData.cc
  PullDatasetController.cc)
target_compile_features(GooStatsSimple PRIVATE cxx_range_for)
target_link_libraries(GooStatsSimple GooStatsKernel)
