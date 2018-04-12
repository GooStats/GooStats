message(STATUS "compiling Reactor modules..")
add_library(reactor_module
  ReactorAnalysisManager.cc
  ReactorDatasetController.cc
  ReactorInputBuilder.cc
  ReactorSpectrumBuilder.cc)
cuda_add_more_library(reactor_gpu_module
  ReactorSpectrumPdf.cu
  NewExpPdf.cu)
target_link_libraries(reactor_gpu_module AllGPU)
