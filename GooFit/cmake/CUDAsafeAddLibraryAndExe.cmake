# Extra CUDA cmake module to make it possible to link separated packages
#
# Both the library and the package that uses the library must use this
# module at the same time.
#
#   Xuefeng Ding http://dingxf.cn
#   Gran Sasso Science Institute
#
#   Copyright (c) 20018
#
#   This code is licensed under the MIT License.  

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

set_property(GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop "")
set_property(GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_TARGETS_prop "")
macro(CUDA_ADD_MORE_EXECUTABLE cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )
  get_property(cache_var GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop)
  set_property(GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS};${cache_var}")
  #message(STATUS "current list: ${CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_executable(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  target_link_libraries(${cuda_target} ${CUDA_LINK_LIBRARIES_KEYWORD}
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()
macro(CUDA_ADD_MORE_LIBRARY cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )
  get_property(cache_var GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop)
  set_property(GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS};${cache_var}")
  get_property(cache_var GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_TARGETS_prop)
  list(APPEND cache_var ${cuda_target})
  set_property(GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_TARGETS_prop "${cache_var}")
  #get_property(cache_var GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop)
  #message(STATUS "current list(${cuda_target}): ${cache_var}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )
    target_link_libraries(${cuda_target} ${CUDA_LINK_LIBRARIES_KEYWORD}
    ${CUDA_LIBRARIES}
    )

  if(CUDA_SEPARABLE_COMPILATION)
    target_link_libraries(${cuda_target} ${CUDA_LINK_LIBRARIES_KEYWORD}
      ${CUDA_cudadevrt_LIBRARY}
      )
  endif()

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()
macro(CUDA_GEN_GPU_LIBRARY cuda_target)
  get_property(this_CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS_prop)
  set(CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS "${this_CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS};${CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS}")
  get_property(this_CUDA_ALL_SEPARABLE_COMPILATION_TARGETS GLOBAL PROPERTY CUDA_ALL_SEPARABLE_COMPILATION_TARGETS_prop)
  list(APPEND CUDA_ALL_SEPARABLE_COMPILATION_TARGETS "${this_CUDA_ALL_SEPARABLE_COMPILATION_TARGETS}")
  foreach(OB ${CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS})
    message(STATUS "${OB}")
  endforeach()
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target}
    "${CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS};${ARGN}")
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target}
    "${options}" "${CUDA_ALL_SEPARABLE_COMPILATION_OBJECTS};${ARGN}")
  add_library(${cuda_target} ${link_file})
  foreach(target ${CUDA_ALL_SEPARABLE_COMPILATION_TARGETS})
    add_dependencies(${cuda_target} ${target})
    message(STATUS "link ${cuda_target} with <${target}>")
    if (TARGET ${target})
      target_link_libraries(${cuda_target} ${target})
    endif()
  endforeach()
  target_link_libraries(${cuda_target} ${CUDA_LINK_LIBRARIES_KEYWORD}
    ${CUDA_cudadevrt_LIBRARY}
    )
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE CXX
    )
endmacro()
