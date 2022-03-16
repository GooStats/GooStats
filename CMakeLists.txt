cmake_minimum_required(VERSION 3.15)
project(GOOSTATS)

add_subdirectory(extern/GooStats/GooFit)
initialize_cuda()

add_library(GooFit::goofit_lib ALIAS goofit_lib)
add_library(GooFit::PDFs ALIAS PDFs)
add_library(GTest::GTest ALIAS gtest_main)
find_package(ROOT CONFIG REQUIRED COMPONENTS Minuit TreePlayer)
if(GOOFIT_DEVICE STREQUAL CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${GOOFIT_ARCH})
  enable_language(CUDA)
endif()

# Options ###
if(MV_CHECK)
  add_definitions("-DMV_CHECK")
endif()
if(NLL_CHECK)
  add_definitions("-DNLL_CHECK")
endif()
if(convolution_CHECK)
  add_definitions("-Dconvolution_CHECK")
endif()
if(RPF_CHECK)
  add_definitions("-DRPF_CHECK")
endif()
if(Quenching_CHECK)
  add_definitions("-DQuenching_CHECK")
endif()
if(Mask_CHECK)
  add_definitions("-DMask_CHECK")
endif()

add_subdirectory(PDFs) # gpu: GooStatsPDFs
add_subdirectory(Kernel) # GooStatsKernel, GooStatsNLLCheck, TestFixture
# also gpu: GooStatsLib_gpu -> linked to GooStatsKernel
add_subdirectory(StatModules) # GooStatsStatModules
add_subdirectory(SimplePackage) # GooStatsSimple
add_subdirectory(src) # GooStatsExperimental

add_library(GooStatsLib INTERFACE)
target_link_libraries(
  GooStatsLib INTERFACE GooStatsKernel GooStatsSimple GooStatsStatModules
                        GooStatsExperimental GooStatsPDFs)
target_compile_features(GooStatsLib INTERFACE cxx_std_11)

add_library(GooStatsLib_NLL INTERFACE)
target_link_libraries(
  GooStatsLib_NLL
  INTERFACE GooStatsKernel GooStatsSimple_NLL GooStatsStatModules
            GooStatsExperimental GooStatsPDFs_NLL)
target_compile_features(GooStatsLib_NLL INTERFACE cxx_std_11)

enable_testing()
add_subdirectory(tests)
