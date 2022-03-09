add_library(
  GooStatsStatModules
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
target_include_directories(
  GooStatsStatModules
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/StatModules/include>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsStatModules PUBLIC GooStatsKernel goofit_base
                                                 ROOT::Tree ROOT::Gpad)
