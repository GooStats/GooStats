cmake_minimum_required(VERSION 3.0)
message(${CMAKE_MODULE_PATH})
# inlcude this first, because the CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS variable will be emptied
include(CUDAsafeAddLibraryAndExe)

find_package(GOOFIT CONFIG REQUIRED)
INITIALIZE_GOOFIT()

include_directories("${PROJECT_SOURCE_DIR}")

cuda_add_more_library(sumPDF SumPdf.cu)
cuda_add_more_executable(exp exponential.cu)

target_link_libraries(exp AllGPU GOOFIT::goofit_lib sumPDF)
target_compile_features(exp PRIVATE cxx_range_for)
