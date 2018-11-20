include_directories("include")
add_executable(fit fit.cc)
target_link_libraries(fit GooStatsLib)
