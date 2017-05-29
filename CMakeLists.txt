cmake_minimum_required (VERSION 2.6)
project (slnn CXX)

# dependency
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(EIGEN3_INCLUDE_DIR "S:/3rdparty/eigen-eigen-9b643c7a9e06")

get_filename_component(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} REALPATH)
message("Eigen3 include dir - " ${EIGEN3_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)


include_directories (${PROJECT_SOURCE_DIR})
include_directories (${EIGEN3_INCLUDE_DIR})



add_subdirectory (symnn)
add_executable (slnn slnn.cpp)

target_link_libraries (slnn symnn)