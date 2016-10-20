INCLUDE_DIRECTORIES(${source_directory})

# crf
set(
    ner_crf_headers
    ner_crf_model.h
    ner_crf_modelhandler.h
)

set(
    ner_crf_libs
    ner_crf_model.cpp
    ner_crf_modelhandler.cpp
)

add_executable(ner_crf ner_crf.cpp 
                      ${ner_crf_headers} ${common_headers}
                      ${ner_crf_libs} ${common_libs}
)

target_link_libraries(ner_crf dynet ${Boost_LIBRARIES})