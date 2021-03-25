goofit_add_library(GooStatsLib_gpu
  GPUManager.cu
  )
target_include_directories(GooStatsLib_gpu PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Kernel/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsLib_gpu PUBLIC GooFit::goofit_lib)

add_library(GooStatsKernel
  RawSpectrumProvider.cc
  ParSyncManager.cc
  BasicSpectrumBuilder.cc
  TextOutputManager.cc
  OutputHelper.cc
  OptionManager.cc
  Module.cc
  AnalysisManager.cc
  DatasetManager.cc
  BasicManagerImpl.cc
  BasicManager.cc
  InputManager.cc
  SpectrumBuilder.cc
  Utility.cc)
target_include_directories(GooStatsKernel PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Kernel/include;${PROJECT_SOURCE_DIR}/PDFs/include;${PROJECT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsKernel PUBLIC GooStatsLib_gpu)

ROOT_GENERATE_DICTIONARY(G__GooStatsNLLCheck ../include/GooStatsNLLCheck.h
  LINKDEF ../include/GooStatsNLLCheckLinkDef.h)
add_library(GooStatsNLLCheck SHARED GooStatsNLLCheck.cc G__GooStatsNLLCheck)
target_include_directories(GooStatsNLLCheck PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Kernel/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(GooStatsNLLCheck PUBLIC ROOT::Core)

add_library(BestFitFixture BestFitFixture.cc)
target_link_libraries(BestFitFixture PUBLIC GooStatsLib_gpu gtest_main)

add_library(NLLFixture NLLCheckFixture.cc)
target_link_libraries(NLLFixture PUBLIC GooStatsLib_gpu gtest_main)

if(NLL_CHECK)
  target_link_libraries(GooStatsKernel PUBLIC GooStatsNLLCheck)
endif()
