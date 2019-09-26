message(STATUS "compiling solarB8 modules..")
add_library(solarB8_module
  solarB8DatasetController.cc
  solarB8InputBuilder.cc
  solarB8SpectrumBuilder.cc)
goofit_add_library(solarB8_gpu_module
  SolarNuOscPdf.cu)
target_link_libraries(solarB8_gpu_module GooStatsPDFs)
