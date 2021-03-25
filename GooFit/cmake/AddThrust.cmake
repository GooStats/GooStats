# Downloads thrust 
#


include(DownloadProject)
message(STATUS "Downloading thrust if needed")
download_project(PROJ                thrust
  		         GIT_REPOSITORY      https://github.com/thrust/thrust.git
                 UPDATE_DISCONNECTED 1
                 QUIET
)

set(THRUST_INCLUDE_DIRS "${thrust_SOURCE_DIR}")

