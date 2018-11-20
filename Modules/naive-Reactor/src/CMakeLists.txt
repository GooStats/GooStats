message(STATUS "compiling Reactor modules..")
add_library(reactor_module
  NMOTest.cc
  ReactorDatasetController.cc
  ReactorInputBuilder.cc
  ReactorSpectrumBuilder.cc)
