include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory(src)
add_subdirectory(Test)
add_executable(reactor reactor.cc)
target_compile_features(reactor PRIVATE cxx_range_for)
target_link_libraries(reactor reactor_module)
target_link_libraries(reactor GooStatsLib)

add_test(NAME    autoTest
           COMMAND autoTest
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
