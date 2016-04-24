INCLUDE_DIRECTORIES(${source_directory})

set(ner_headers
    bilstmmodel4ner.hpp

)

ADD_EXECUTABLE(ner ner.cc ${common_headers} ${common_libs} ${ner_headers})

target_link_libraries(ner cnn ${Boost_LIBRARIES})       