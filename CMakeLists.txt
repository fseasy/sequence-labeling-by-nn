cmake_minimum_required (VERSION 2.6)
project (slnn CXX)

include_directories (${PROJECT_SOURCE_DIR})

add_subdirectory (symnn)
add_executable (slnn slnn.cpp)

target_link_libraries (slnn symnn)