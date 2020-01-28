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
goofit_add_library(GooStatsLib_gpu
  GPUManager.cu
  )
target_link_libraries(GooStatsKernel GooStatsLib_gpu)

ROOT_GENERATE_DICTIONARY(G__GooStatsNLLCheck ../include/GooStatsNLLCheck.h
  LINKDEF ../include/GooStatsNLLCheckLinkDef.h)
add_library(GooStatsNLLCheck SHARED GooStatsNLLCheck.cc G__GooStatsNLLCheck)
target_link_libraries(GooStatsNLLCheck ${ROOT_LIBRARIES})

add_library(BestFitFixture BestFitFixture.cc)
target_link_libraries(BestFitFixture GTest::GTest ${ROOT_LIBRARIES})

add_library(NLLFixture NLLCheckFixture.cc)
target_link_libraries(NLLFixture GooStatsNLLCheck GTest::GTest ${ROOT_LIBRARIES})
#INSTALL(FILES ${ROOT_DICT_OUTPUT_DIR}/libGooStatsNLLCheck_rdict.pcm DESTINATION
#  ${LIBRARY_OUTPUT_PATH})

if(NLL_CHECK)
  target_link_libraries(GooStatsKernel GooStatsNLLCheck)
endif()
