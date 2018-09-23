add_library(GooStatsLib
  AnalysisManager.cc
  BasicManager.cc
  BasicManagerImpl.cc
  BasicSpectrumBuilder.cc
  ConfigsetManager.cc
  DatasetManager.cc
  SimpleDatasetController.cc
  PullDatasetController.cc
  SimpleInputBuilder.cc
  SimpleOptionParser.cc
  SimplePlotManager.cc
  SpectrumBuilder.cc
  InputManager.cc
  OutputManager.cc
  SimpleOutputBuilder.cc
  OutputHelper.cc
  BatchOutputManager.cc
  PlotManager.cc
  ParSyncManager.cc
  RawSpectrumProvider.cc
  TextOutputManager.cc)
target_compile_features(GooStatsLib PRIVATE cxx_range_for)
cuda_add_library(GooStatsLib_gpu
  GPUManager.cu
  )

ROOT_GENERATE_DICTIONARY(G__GooStatsNLLCheck ../include/GooStatsNLLCheck.h
  LINKDEF ../include/GooStatsNLLCheckLinkDef.h)
add_library(GooStatsNLLCheck SHARED GooStatsNLLCheck.cc G__GooStatsNLLCheck.cxx)
target_link_libraries(GooStatsNLLCheck ${ROOT_LIBRARIES})

add_library(TestFixture BestFitFixture.cc NLLCheckFixture.cc)
target_link_libraries(TestFixture ${ROOT_LIBRARIES})
#INSTALL(FILES ${ROOT_DICT_OUTPUT_DIR}/libGooStatsNLLCheck_rdict.pcm DESTINATION
#  ${LIBRARY_OUTPUT_PATH})

if(NLL_CHECK)
  target_link_libraries(GooStatsLib GooStatsNLLCheck)
endif()
