message(STATUS "compiling Juno modules..")
add_library(juno_module
  JunoAnalysisManager.cc
  JunoDatasetController.cc
  JunoInputBuilder.cc
  JunoSpectrumBuilder.cc)
cuda_add_more_library(juno_gpu_module
  ReactorSpectrumPdf.cu
  NewExpPdf.cu)
target_link_libraries(juno_gpu_module AllGPU)
