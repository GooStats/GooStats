add_library(GooStatsKernel
  RawSpectrumProvider.cc
  ParSyncManager.cc
  NLLCheckFixture.cc
  GooStatsNLLCheck.cc
  BasicSpectrumBuilder.cc
  TextOutputManager.cc
  OutputHelper.cc
  OptionManager.cc
  Module.cc
  BestFitFixture.cc
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
add_library(GooStatsNLLCheck SHARED GooStatsNLLCheck.cc G__GooStatsNLLCheck.cxx)
target_link_libraries(GooStatsNLLCheck ${ROOT_LIBRARIES})

add_library(TestFixture BestFitFixture.cc NLLCheckFixture.cc)
target_link_libraries(TestFixture ${ROOT_LIBRARIES})
#INSTALL(FILES ${ROOT_DICT_OUTPUT_DIR}/libGooStatsNLLCheck_rdict.pcm DESTINATION
#  ${LIBRARY_OUTPUT_PATH})

if(NLL_CHECK)
  target_link_libraries(GooStatsKernel GooStatsNLLCheck)
endif()
