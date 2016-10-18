INCLUDE_DIRECTORIES(${source_directory})

# crf dc
set(
    ner_crf_dc_headers
    ner_crf_dc_model.h
    ner_crf_dc_modelhandler.h
)

set(
    ner_crf_dc_libs
    ner_crf_dc_model.cpp
    ner_crf_dc_modelhandler.cpp
)

add_executable(ner_crf_dc ner_crf_dc.cpp 
                      ${ner_crf_dc_headers} ${common_headers}
                      ${ner_crf_dc_libs} ${common_libs}
)

target_link_libraries(ner_crf_dc dynet ${Boost_LIBRARIES})