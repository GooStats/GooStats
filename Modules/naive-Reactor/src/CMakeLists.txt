message(STATUS "compiling Reactor modules..")
add_library(juno_module
  ReactorAnalysisManager.cc
  ReactorDatasetController.cc
  ReactorInputBuilder.cc
  ReactorSpectrumBuilder.cc)
cuda_add_more_library(juno_gpu_module
  ReactorSpectrumPdf.cu
  NewExpPdf.cu)
target_link_libraries(juno_gpu_module AllGPU)
