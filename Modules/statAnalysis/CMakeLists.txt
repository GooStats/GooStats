include_directories("include")
add_executable(stat stat.cc)
target_link_libraries(stat GooStatsLib)
