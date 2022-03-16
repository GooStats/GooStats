add_library(
  GooStatsSimpleObj OBJECT
  SimpleDatasetController.cc
  SimpleInputBuilder.cc
  SimpleOutputBuilder.cc
  SimpleSpectrumBuilder.cc
  SimplePlotManager.cc
  PrepareData.cc
  PullDatasetController.cc
  SimpleOptionParser.cc
  ../include/SimpleOptionParser.h)
add_sanitizers(GooStatsSimpleObj)
target_include_directories(
  GooStatsSimpleObj
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/SimplePackage/include>"
         "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/PDFs/include>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(
  GooStatsSimpleObj PUBLIC GooStatsKernel GooStatsStatModules
                           GooStatsExperimental goofit_base ROOT::Gpad)

add_library(GooStatsSimple)
target_link_libraries(GooStatsSimple PUBLIC GooStatsSimpleObj GooStatsPDFs
                                            GooStatsStatModules ROOT::Gpad)

add_library(GooStatsSimple_NLL)
target_link_libraries(
  GooStatsSimple_NLL PUBLIC GooStatsSimpleObj GooStatsPDFs_NLL
                            GooStatsStatModules ROOT::Gpad)
