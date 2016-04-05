INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(postagger postagger.cc utf8processing.hpp)
target_link_libraries(postagger cnn ${Boost_LIBRARIES})
