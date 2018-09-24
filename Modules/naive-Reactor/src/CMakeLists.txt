message(STATUS "compiling Reactor modules..")
add_library(reactor_module
  ReactorAnalysisManager.cc
  ReactorDatasetController.cc
  ReactorInputBuilder.cc
  ReactorSpectrumBuilder.cc)
