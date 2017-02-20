INCLUDE_DIRECTORIES(${source_directory})

set(ner_headers
    bilstmmodel4ner.hpp
)


ADD_EXECUTABLE(ner_cl ner_cl.cpp ${common_headers} ${common_libs} ${ner_headers})

target_link_libraries(ner_cl dynet ${Boost_LIBRARIES})       
