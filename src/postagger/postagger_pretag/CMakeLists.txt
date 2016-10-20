INCLUDE_DIRECTORIES(${source_directory})

# pretag
set(postagger_headers
    bilstmmodel4tagging.hpp
)
ADD_EXECUTABLE(postagger_pretag postagger.cpp ${common_headers} ${common_libs} ${postagger_headers})
target_link_libraries(postagger_pretag dynet ${Boost_LIBRARIES})
