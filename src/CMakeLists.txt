add_library(GooStatsLib
  AnalysisManager.cc
  BasicManager.cc
  BasicManagerImpl.cc
  BasicSpectrumBuilder.cc
  ConfigsetManager.cc
  DatasetManager.cc
  SimpleDatasetController.cc
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

