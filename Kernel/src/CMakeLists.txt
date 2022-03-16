goofit_add_library(GooStatsLib_gpu GPUManager.cu)
target_include_directories(
  GooStatsLib_gpu
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Kernel/include>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsLib_gpu PUBLIC ROOT::Core)

add_library(
  GooStatsKernel
  RawSpectrumProvider.cc
  BasicSpectrumBuilder.cc
  TextOutputManager.cc
  OutputHelper.cc
  OptionManager.cc
  Module.cc
  AnalysisManager.cc
  DatasetManager.cc
  BasicManager.cc
  InputManager.cc
  SpectrumBuilder.cc
  Utility.cc
  DatasetController.cpp
  ../include/IOptionParser.h
  Database.cc
  ../include/Database.h
  ../include/MultiComponentDatasetController.h
  MultiComponetDatasetController.cc)
add_sanitizers(GooStatsKernel)

target_include_directories(
  GooStatsKernel
  PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Kernel/include;${PROJECT_SOURCE_DIR}/PDFs/include;${PROJECT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsKernel PUBLIC GooStatsLib_gpu goofit_base)

add_library(BestFitFixture BestFitFixture.cc)
add_sanitizers(BestFitFixture)
target_link_libraries(BestFitFixture PUBLIC GooStatsLib_gpu gtest_main
                                            ROOT::TreePlayer)

add_library(NLLFixture NLLCheckFixture.cc)
add_sanitizers(NLLFixture)
target_link_libraries(NLLFixture PUBLIC GooStatsLib_gpu gtest_main
                                        GooStatsNLLCheck)
