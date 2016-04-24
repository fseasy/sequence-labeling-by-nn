INCLUDE_DIRECTORIES(${source_directory})

set(postagger_headers
    bilstmmodel4tagging.hpp
)

ADD_EXECUTABLE(postagger postagger.cc ${common_headers} ${common_libs} ${postagger_headers})


target_link_libraries(postagger cnn ${Boost_LIBRARIES})