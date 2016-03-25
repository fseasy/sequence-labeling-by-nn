INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(postagger postagger.cc)
target_link_libraries(postagger cnn ${Boost_LIBRARIES})
