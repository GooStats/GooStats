add_library(GooStatsStatModules
  BatchOutputManager.cc
  ContourManager.cc
  CorrelationManager.cc
  DiscoveryTest.cc
  GSFitManager.cc
  ScanPar.cc
  SimpleFit.cc
  StatModule.cc
  OutputManager.cc
  PlotManager.cc
  ToyMCLikelihoodEvaluator.cc)
target_link_libraries(GooStatsStatModules GooStatsKernel)
