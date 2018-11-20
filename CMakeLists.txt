cmake_minimum_required(VERSION 3.4)
project(GOOSTATS CXX)

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
message(STATUS ${LOC_PATH})
if(EXISTS ${LOC_PATH})
    message(FATAL_ERROR "You cannot build in the source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory.")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-Werror -Wall -Wextra -Werror=return-type ${CMAKE_CXX_FLAGS}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
message(STATUS ${CMAKE_MODULE_PATH})
include(CUDAsafeAddLibraryAndExe)
################################## google test
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
##################################

### GOOFIT
find_package(GOOFIT CONFIG REQUIRED)
INITIALIZE_GOOFIT()
#find_package(ROOT REQUIRED COMPONENTS RIO Net)
include(${ROOT_USE_FILE}/../modules/RootNewMacros.cmake)

set(GOOSTATS_NO_WARNING ON)
if(GOOSTATS_NO_WARNING AND ${CUDA_VERSION} VERSION_GREATER 7.0)
  message(STATUS "GooStats silent_thrust_warning enabled!")
set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-I${PROJECT_SOURCE_DIR}/silent_thrust_warning/include")
include_directories("${PROJECT_SOURCE_DIR}/silent_thrust_warning/include")
endif()

include_directories("${PROJECT_SOURCE_DIR}/include")

### Options ###
if(GOOSTATS_SEPARATE_COMP)
    add_definitions("-DSEPARABLE")
endif()
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


include_directories("Kernel/include")
include_directories("PDFs/include")
add_subdirectory(PDFs) # gpu: GooStatsPDFs
add_subdirectory(Kernel) # GooStatsKernel, GooStatsNLLCheck, TestFixture
# also gpu: GooStatsLib_gpu -> linked to GooStatsKernel
include_directories("StatModules/include")
add_subdirectory(StatModules) # GooStatsStatModules
include_directories("SimplePackage/include")
add_subdirectory(SimplePackage) # GooStatsSimple
include_directories("include")
add_subdirectory(src) # GooStatsExperimental

add_library(GooStatsLib)
target_link_libraries(GooStatsLib GooStatsKernel GooStatsSimple
  GooStatsStatModules GooStatsExperimental GooStatsLib_gpu
  GooStatsPDFs GOOFIT::goofit_lib AllGPU)


add_subdirectory(Modules)

cuda_gen_gpu_library(AllGPU)
target_link_libraries(AllGPU GOOFIT::goofit_lib GooStatsPDFs)
