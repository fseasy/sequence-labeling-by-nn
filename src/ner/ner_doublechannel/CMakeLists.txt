INCLUDE_DIRECTORIES(${source_directory})

set(
    ner_dc_headers
    ner_dc_model.h
    ner_dc_modelhandler.h
)

set(
    ner_dc_libs
    ner_dc_model.cpp
    ner_dc_modelhandler.cpp
)

add_executable(ner_dc ner_doublechannel.cpp 
                      ${ner_dc_headers} ${common_headers}
                      ${ner_dc_libs} ${common_libs}
)

target_link_libraries(ner_dc cnn ${Boost_LIBRARIES})