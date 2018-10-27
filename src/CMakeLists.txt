add_library(GooStatsExperimental
  DeclareModule.cc
  ModuleFactory.cc
  ModuleManager.cc)
target_compile_features(GooStatsExperimental PRIVATE cxx_range_for)
